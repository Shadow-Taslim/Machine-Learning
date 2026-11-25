import { mulberry32 } from './csv';

// ============================================================================
// 1. FrozenLake-v1 (4x4) Q-Learning
// ============================================================================

export const FROZEN_LAKE_GRID = ['SFFF', 'FHFH', 'FFFH', 'HFFG'];
export const FROZEN_ARROWS = ['←', '↓', '→', '↑'] as const;

function tileAt(r: number, c: number): string {
  return FROZEN_LAKE_GRID[r][c];
}

function moveDelta(r: number, c: number, action: number): [number, number] {
  let nr = r;
  let nc = c;
  if (action === 0) nc = Math.max(0, c - 1); // left
  else if (action === 1) nr = Math.min(3, r + 1); // down
  else if (action === 2) nc = Math.min(3, c + 1); // right
  else if (action === 3) nr = Math.max(0, r - 1); // up
  return [nr, nc];
}

export function stepFrozenLake(
  state: number,
  action: number,
  isSlippery: boolean,
  rng: () => number
): { nextState: number; reward: number; done: boolean; actualAction: number } {
  const r = Math.floor(state / 4);
  const c = state % 4;
  let actualAction = action;
  if (isSlippery) {
    const roll = rng();
    if (roll < 1 / 3) actualAction = (action + 3) % 4;
    else if (roll < 2 / 3) actualAction = action;
    else actualAction = (action + 1) % 4;
  }
  const [nr, nc] = moveDelta(r, c, actualAction);
  const tile = tileAt(nr, nc);
  const nextState = nr * 4 + nc;
  if (tile === 'G') return { nextState, reward: 1, done: true, actualAction };
  if (tile === 'H') return { nextState, reward: 0, done: true, actualAction };
  return { nextState, reward: 0, done: false, actualAction };
}

export interface QLearningResult {
  Q: number[][]; // 16 x 4
  movingAvg: number[];
  epsHistory: number[];
  randomSuccessRate: number;
  trainedSuccessRate: number;
}

export function trainFrozenLakeQLearning(options: {
  episodes: number;
  alpha: number;
  gamma: number;
  decayRate: number;
  isSlippery: boolean;
  seed?: number;
}): QLearningResult {
  const rng = mulberry32(options.seed ?? 42);
  const Q: number[][] = Array.from({ length: 16 }, () => [0, 0, 0, 0]);
  const rewards: number[] = [];
  const epsHistory: number[] = [];

  let epsilon = 1.0;
  const minEpsilon = 0.01;

  for (let ep = 0; ep < options.episodes; ep++) {
    let state = 0;
    let totalReward = 0;
    for (let step = 0; step < 100; step++) {
      let action = 0;
      if (rng() < epsilon) {
        action = Math.floor(rng() * 4);
      } else {
        const qRow = Q[state];
        let bestA = 0;
        for (let a = 1; a < 4; a++) {
          if (qRow[a] > qRow[bestA]) bestA = a;
        }
        action = bestA;
      }

      const { nextState, reward, done } = stepFrozenLake(
        state,
        action,
        options.isSlippery,
        rng
      );
      const maxNextQ = done ? 0 : Math.max(...Q[nextState]);
      const tdTarget = reward + options.gamma * maxNextQ;
      Q[state][action] += options.alpha * (tdTarget - Q[state][action]);

      totalReward += reward;
      state = nextState;
      if (done) break;
    }
    rewards.push(totalReward);
    epsHistory.push(epsilon);
    epsilon = Math.max(minEpsilon, epsilon * options.decayRate);
  }

  const window = Math.min(200, Math.max(20, Math.floor(options.episodes / 15)));
  const movingAvg: number[] = [];
  let rolling = 0;
  for (let i = 0; i < rewards.length; i++) {
    rolling += rewards[i];
    if (i >= window) rolling -= rewards[i - window];
    if (i >= window - 1) {
      movingAvg.push(rolling / window);
    }
  }

  // Evaluate Random vs Greedy over 500 episodes
  const evalEpisodes = 500;
  let randWins = 0;
  let trainedWins = 0;
  for (let ep = 0; ep < evalEpisodes; ep++) {
    let s1 = 0;
    for (let t = 0; t < 100; t++) {
      const a = Math.floor(rng() * 4);
      const res = stepFrozenLake(s1, a, options.isSlippery, rng);
      s1 = res.nextState;
      if (res.done) {
        randWins += res.reward;
        break;
      }
    }
    let s2 = 0;
    for (let t = 0; t < 100; t++) {
      const qRow = Q[s2];
      let bestA = 0;
      for (let a = 1; a < 4; a++) if (qRow[a] > qRow[bestA]) bestA = a;
      const res = stepFrozenLake(s2, bestA, options.isSlippery, rng);
      s2 = res.nextState;
      if (res.done) {
        trainedWins += res.reward;
        break;
      }
    }
  }

  return {
    Q,
    movingAvg,
    epsHistory,
    randomSuccessRate: randWins / evalEpisodes,
    trainedSuccessRate: trainedWins / evalEpisodes,
  };
}

export interface FrozenLakeStep {
  step: number;
  state: number;
  row: number;
  col: number;
  action: number;
  actualAction: number;
  reward: number;
  done: boolean;
}

export function rolloutFrozenLake(
  Q: number[][],
  isSlippery: boolean,
  seed = 7
): FrozenLakeStep[] {
  const rng = mulberry32(seed);
  let state = 0;
  const steps: FrozenLakeStep[] = [
    {
      step: 0,
      state: 0,
      row: 0,
      col: 0,
      action: -1,
      actualAction: -1,
      reward: 0,
      done: false,
    },
  ];
  for (let t = 1; t <= 60; t++) {
    const qRow = Q[state];
    let bestA = 0;
    for (let a = 1; a < 4; a++) if (qRow[a] > qRow[bestA]) bestA = a;
    const res = stepFrozenLake(state, bestA, isSlippery, rng);
    steps.push({
      step: t,
      state: res.nextState,
      row: Math.floor(res.nextState / 4),
      col: res.nextState % 4,
      action: bestA,
      actualAction: res.actualAction,
      reward: res.reward,
      done: res.done,
    });
    state = res.nextState;
    if (res.done) break;
  }
  return steps;
}

// ============================================================================
// 2. CartPole-v1 Physics & DQN (4 -> 24 -> 24 -> 2 MLP + Replay + Target Net)
// ============================================================================

export type CartPoleState = [number, number, number, number]; // x, xDot, theta, thetaDot

export const CARTPOLE_X_THRESHOLD = 2.4;
export const CARTPOLE_THETA_THRESHOLD = (12 * 2 * Math.PI) / 360;

export function stepCartPole(
  state: CartPoleState,
  action: number // 0 = left, 1 = right
): { nextState: CartPoleState; reward: number; done: boolean } {
  const gravity = 9.8;
  const masscart = 1.0;
  const masspole = 0.1;
  const totalMass = masscart + masspole;
  const length = 0.5;
  const polemassLength = masspole * length;
  const forceMag = 10.0;
  const tau = 0.02;

  let [x, xDot, theta, thetaDot] = state;
  const force = action === 1 ? forceMag : -forceMag;
  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);

  const temp = (force + polemassLength * thetaDot * thetaDot * sinTheta) / totalMass;
  const thetaAcc =
    (gravity * sinTheta - cosTheta * temp) /
    (length * (4.0 / 3.0 - (masspole * cosTheta * cosTheta) / totalMass));
  const xAcc = temp - (polemassLength * thetaAcc * cosTheta) / totalMass;

  x = x + tau * xDot;
  xDot = xDot + tau * xAcc;
  theta = theta + tau * thetaDot;
  thetaDot = thetaDot + tau * thetaAcc;

  const done =
    x < -CARTPOLE_X_THRESHOLD ||
    x > CARTPOLE_X_THRESHOLD ||
    theta < -CARTPOLE_THETA_THRESHOLD ||
    theta > CARTPOLE_THETA_THRESHOLD;

  return {
    nextState: [x, xDot, theta, thetaDot],
    reward: done ? 0 : 1,
    done,
  };
}

export function resetCartPole(rng: () => number): CartPoleState {
  return [
    (rng() - 0.5) * 0.1,
    (rng() - 0.5) * 0.1,
    (rng() - 0.5) * 0.1,
    (rng() - 0.5) * 0.1,
  ];
}

interface MLP {
  w1: number[][]; // 4 x H
  b1: number[]; // H
  w2: number[][]; // H x H
  b2: number[]; // H
  w3: number[][]; // H x 2
  b3: number[]; // 2
}

function createMLP(hidden: number, rng: () => number): MLP {
  const randMat = (r: number, c: number) => {
    const scale = Math.sqrt(2 / r);
    return Array.from({ length: r }, () =>
      Array.from({ length: c }, () => (rng() * 2 - 1) * scale)
    );
  };
  return {
    w1: randMat(4, hidden),
    b1: Array(hidden).fill(0),
    w2: randMat(hidden, hidden),
    b2: Array(hidden).fill(0),
    w3: randMat(hidden, 2),
    b3: [0, 0],
  };
}

function cloneMLP(net: MLP): MLP {
  return {
    w1: net.w1.map((r) => [...r]),
    b1: [...net.b1],
    w2: net.w2.map((r) => [...r]),
    b2: [...net.b2],
    w3: net.w3.map((r) => [...r]),
    b3: [...net.b3],
  };
}

function forwardMLP(net: MLP, x: CartPoleState) {
  const h = net.b1.length;
  const z1 = Array(h).fill(0);
  const a1 = Array(h).fill(0);
  for (let j = 0; j < h; j++) {
    let s = net.b1[j];
    for (let i = 0; i < 4; i++) s += x[i] * net.w1[i][j];
    z1[j] = s;
    a1[j] = s > 0 ? s : 0;
  }

  const z2 = Array(h).fill(0);
  const a2 = Array(h).fill(0);
  for (let j = 0; j < h; j++) {
    let s = net.b2[j];
    for (let i = 0; i < h; i++) s += a1[i] * net.w2[i][j];
    z2[j] = s;
    a2[j] = s > 0 ? s : 0;
  }

  const out = [net.b3[0], net.b3[1]];
  for (let j = 0; j < 2; j++) {
    for (let i = 0; i < h; i++) out[j] += a2[i] * net.w3[i][j];
  }
  return { a1, z1, a2, z2, out };
}

interface Transition {
  s: CartPoleState;
  a: number;
  r: number;
  ns: CartPoleState;
  done: boolean;
}

export interface DQNResult {
  returnsHistory: number[];
  movingAvg: number[];
  epsHistory: number[];
  randomAvgReturn: number;
  trainedAvgReturn: number;
  bestEpisodeTrajectory: { state: CartPoleState; action: number; step: number }[];
  policyNet: MLP;
}

export function trainCartPoleDQN(options: {
  episodes?: number;
  gamma?: number;
  lr?: number;
  decayRate?: number;
  seed?: number;
} = {}): DQNResult {
  const episodes = options.episodes ?? 220;
  const gamma = options.gamma ?? 0.99;
  const lr = options.lr ?? 0.0025;
  const decayRate = options.decayRate ?? 0.975;
  const rng = mulberry32(options.seed ?? 42);

  const hidden = 24;
  const policyNet = createMLP(hidden, rng);
  let targetNet = cloneMLP(policyNet);

  const buffer: Transition[] = [];
  const maxBuffer = 6000;
  const batchSize = 32;

  const returnsHistory: number[] = [];
  const epsHistory: number[] = [];
  let epsilon = 1.0;
  const minEpsilon = 0.02;
  let globalSteps = 0;

  const trainBatch = () => {
    if (buffer.length < batchSize) return;
    // Accumulate gradients over batch
    const gw1 = policyNet.w1.map((r) => r.map(() => 0));
    const gb1 = policyNet.b1.map(() => 0);
    const gw2 = policyNet.w2.map((r) => r.map(() => 0));
    const gb2 = policyNet.b2.map(() => 0);
    const gw3 = policyNet.w3.map((r) => r.map(() => 0));
    const gb3 = [0, 0];

    for (let b = 0; b < batchSize; b++) {
      const tr = buffer[Math.floor(rng() * buffer.length)];
      const fwd = forwardMLP(policyNet, tr.s);
      const nextQ = forwardMLP(targetNet, tr.ns).out;
      const targetVal = tr.done
        ? tr.r
        : tr.r + gamma * Math.max(nextQ[0], nextQ[1]);

      const err = Math.max(-1.5, Math.min(1.5, fwd.out[tr.a] - targetVal));
      const dOut = [0, 0];
      dOut[tr.a] = err / batchSize;

      gb3[tr.a] += dOut[tr.a];
      const da2 = Array(hidden).fill(0);
      for (let i = 0; i < hidden; i++) {
        gw3[i][tr.a] += fwd.a2[i] * dOut[tr.a];
        da2[i] = policyNet.w3[i][tr.a] * dOut[tr.a];
      }

      const dz2 = da2.map((v, i) => (fwd.z2[i] > 0 ? v : 0));
      const da1 = Array(hidden).fill(0);
      for (let j = 0; j < hidden; j++) {
        gb2[j] += dz2[j];
        for (let i = 0; i < hidden; i++) {
          gw2[i][j] += fwd.a1[i] * dz2[j];
          da1[i] += policyNet.w2[i][j] * dz2[j];
        }
      }

      const dz1 = da1.map((v, i) => (fwd.z1[i] > 0 ? v : 0));
      for (let j = 0; j < hidden; j++) {
        gb1[j] += dz1[j];
        for (let i = 0; i < 4; i++) {
          gw1[i][j] += tr.s[i] * dz1[j];
        }
      }
    }

    for (let i = 0; i < 4; i++)
      for (let j = 0; j < hidden; j++) policyNet.w1[i][j] -= lr * gw1[i][j];
    for (let j = 0; j < hidden; j++) policyNet.b1[j] -= lr * gb1[j];
    for (let i = 0; i < hidden; i++)
      for (let j = 0; j < hidden; j++) policyNet.w2[i][j] -= lr * gw2[i][j];
    for (let j = 0; j < hidden; j++) policyNet.b2[j] -= lr * gb2[j];
    for (let i = 0; i < hidden; i++)
      for (let j = 0; j < 2; j++) policyNet.w3[i][j] -= lr * gw3[i][j];
    for (let j = 0; j < 2; j++) policyNet.b3[j] -= lr * gb3[j];
  };

  for (let ep = 0; ep < episodes; ep++) {
    let state = resetCartPole(rng);
    let epReturn = 0;
    for (let t = 0; t < 300; t++) {
      let action = 0;
      if (rng() < epsilon) {
        action = rng() < 0.5 ? 0 : 1;
      } else {
        const q = forwardMLP(policyNet, state).out;
        action = q[1] > q[0] ? 1 : 0;
      }

      const res = stepCartPole(state, action);
      epReturn += 1;
      if (buffer.length >= maxBuffer) buffer.shift();
      buffer.push({
        s: state,
        a: action,
        r: res.done ? 0 : 1,
        ns: res.nextState,
        done: res.done,
      });

      globalSteps++;
      if (globalSteps % 2 === 0) trainBatch();
      if (globalSteps % 200 === 0) targetNet = cloneMLP(policyNet);

      state = res.nextState;
      if (res.done) break;
    }
    returnsHistory.push(epReturn);
    epsHistory.push(epsilon);
    epsilon = Math.max(minEpsilon, epsilon * decayRate);
  }

  const window = 15;
  const movingAvg: number[] = [];
  let sum = 0;
  for (let i = 0; i < returnsHistory.length; i++) {
    sum += returnsHistory[i];
    if (i >= window) sum -= returnsHistory[i - window];
    if (i >= window - 1) movingAvg.push(sum / window);
  }

  // Evaluate random vs trained
  let randSum = 0;
  let trainedSum = 0;
  let bestTraj: { state: CartPoleState; action: number; step: number }[] = [];

  for (let ep = 0; ep < 15; ep++) {
    let sRand = resetCartPole(rng);
    let stepsRand = 0;
    for (let t = 0; t < 500; t++) {
      const res = stepCartPole(sRand, rng() < 0.5 ? 0 : 1);
      stepsRand++;
      sRand = res.nextState;
      if (res.done) break;
    }
    randSum += stepsRand;

    let sTrained = resetCartPole(rng);
    const traj: { state: CartPoleState; action: number; step: number }[] = [];
    for (let t = 0; t < 500; t++) {
      const q = forwardMLP(policyNet, sTrained).out;
      const action = q[1] > q[0] ? 1 : 0;
      traj.push({ state: [...sTrained] as CartPoleState, action, step: t });
      const res = stepCartPole(sTrained, action);
      sTrained = res.nextState;
      if (res.done) break;
    }
    trainedSum += traj.length;
    if (traj.length > bestTraj.length) bestTraj = traj;
  }

  return {
    returnsHistory,
    movingAvg,
    epsHistory,
    randomAvgReturn: randSum / 15,
    trainedAvgReturn: trainedSum / 15,
    bestEpisodeTrajectory: bestTraj,
    policyNet,
  };
}

export function dqnGreedyAction(policyNet: MLP, state: CartPoleState): number {
  const q = forwardMLP(policyNet, state).out;
  return q[1] > q[0] ? 1 : 0;
}
