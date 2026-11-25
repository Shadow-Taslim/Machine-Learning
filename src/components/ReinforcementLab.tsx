import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  CARTPOLE_X_THRESHOLD,
  FROZEN_ARROWS,
  FROZEN_LAKE_GRID,
  rolloutFrozenLake,
  trainCartPoleDQN,
  trainFrozenLakeQLearning,
} from '../lib/rl';
import { MetricTile, MultiLineChart } from './Charts';

export const ReinforcementLab: React.FC = () => {
  const [tab, setTab] = useState<'frozenlake' | 'cartpole'>('frozenlake');

  // FrozenLake Q-Learning State
  const [isSlippery, setIsSlippery] = useState(true);
  const [episodes, setEpisodes] = useState(4000);
  const [alpha, setAlpha] = useState(0.1);
  const [gamma, setGamma] = useState(0.99);
  const [rolloutSeed, setRolloutSeed] = useState(7);
  const [stepIndex, setStepIndex] = useState(0);
  const [playingLake, setPlayingLake] = useState(false);

  const lakeResult = useMemo(
    () =>
      trainFrozenLakeQLearning({
        episodes,
        alpha,
        gamma,
        decayRate: isSlippery ? 0.999 : 0.9985,
        isSlippery,
        seed: 42,
      }),
    [episodes, alpha, gamma, isSlippery]
  );

  const lakeSteps = useMemo(
    () => rolloutFrozenLake(lakeResult.Q, isSlippery, rolloutSeed),
    [lakeResult.Q, isSlippery, rolloutSeed]
  );

  useEffect(() => {
    setStepIndex(0);
    setPlayingLake(false);
  }, [lakeSteps]);

  useEffect(() => {
    if (!playingLake) return;
    const timer = window.setInterval(() => {
      setStepIndex((prev) => {
        if (prev >= lakeSteps.length - 1) {
          setPlayingLake(false);
          return prev;
        }
        return prev + 1;
      });
    }, 320);
    return () => window.clearInterval(timer);
  }, [playingLake, lakeSteps.length]);

  // CartPole DQN State
  const [dqnEpisodes, setDqnEpisodes] = useState(220);
  const [dqnSeed, setDqnSeed] = useState(42);
  const [cartStep, setCartStep] = useState(0);
  const [playingCart, setPlayingCart] = useState(true);
  const cartCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const dqnResult = useMemo(
    () => trainCartPoleDQN({ episodes: dqnEpisodes, seed: dqnSeed }),
    [dqnEpisodes, dqnSeed]
  );

  useEffect(() => {
    setCartStep(0);
  }, [dqnResult]);

  useEffect(() => {
    if (tab !== 'cartpole' || !playingCart) return;
    const traj = dqnResult.bestEpisodeTrajectory;
    if (traj.length === 0) return;
    const timer = window.setInterval(() => {
      setCartStep((prev) => (prev + 1) % traj.length);
    }, 28);
    return () => window.clearInterval(timer);
  }, [tab, playingCart, dqnResult]);

  useEffect(() => {
    if (tab !== 'cartpole' || !cartCanvasRef.current) return;
    const canvas = cartCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const traj = dqnResult.bestEpisodeTrajectory;
    const frame = traj[Math.min(cartStep, traj.length - 1)];
    if (!frame) return;

    const [x, , theta] = frame.state;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#F8FAFC';
    ctx.fillRect(0, 0, w, h);

    const trackY = h * 0.74;
    const worldWidth = CARTPOLE_X_THRESHOLD * 2.2;
    const scale = w / worldWidth;
    const cartX = w / 2 + x * scale;

    // Track line & boundary markers
    ctx.strokeStyle = '#CBD5E1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(24, trackY);
    ctx.lineTo(w - 24, trackY);
    ctx.stroke();

    const boundL = w / 2 - CARTPOLE_X_THRESHOLD * scale;
    const boundR = w / 2 + CARTPOLE_X_THRESHOLD * scale;
    ctx.strokeStyle = '#FDA4AF';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(boundL, trackY - 30);
    ctx.lineTo(boundL, trackY + 20);
    ctx.moveTo(boundR, trackY - 30);
    ctx.lineTo(boundR, trackY + 20);
    ctx.stroke();
    ctx.setLineDash([]);

    // Cart
    const cartW = 62;
    const cartH = 30;
    ctx.fillStyle = '#0F172A';
    ctx.fillRect(cartX - cartW / 2, trackY - cartH / 2, cartW, cartH);

    // Force arrow
    ctx.fillStyle = '#2563EB';
    ctx.font = '600 12px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(
      frame.action === 1 ? 'PUSH →' : '← PUSH',
      cartX,
      trackY + 34
    );

    // Pole
    const poleLen = 110;
    const tipX = cartX + Math.sin(theta) * poleLen;
    const tipY = trackY - Math.cos(theta) * poleLen;

    ctx.strokeStyle = '#D97706';
    ctx.lineWidth = 7;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cartX, trackY - 4);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    // Pivot pin
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(cartX, trackY - 4, 4, 0, Math.PI * 2);
    ctx.fill();
  }, [tab, cartStep, dqnResult]);

  const currentLakeFrame = lakeSteps[Math.min(stepIndex, lakeSteps.length - 1)];
  const currentCartFrame =
    dqnResult.bestEpisodeTrajectory[
      Math.min(cartStep, Math.max(0, dqnResult.bestEpisodeTrajectory.length - 1))
    ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-slate-900">
            Reinforcement Learning — Q-Learning & Deep Q-Network
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            FrozenLake-v1 tabular Bellman updates · CartPole-v1 continuous physics + MLP DQN
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg text-xs">
          <button
            type="button"
            onClick={() => setTab('frozenlake')}
            className={`px-3 py-1.5 font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              tab === 'frozenlake'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            FrozenLake Q-Learning
          </button>
          <button
            type="button"
            onClick={() => setTab('cartpole')}
            className={`px-3 py-1.5 font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              tab === 'cartpole'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            DQN CartPole-v1
          </button>
        </div>
      </div>

      {tab === 'frozenlake' ? (
        <div className="space-y-6">
          <div className="bg-white border border-slate-200 rounded-lg p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
            <div className="flex flex-wrap items-center gap-5">
              <label className="inline-flex items-center gap-2 font-medium text-slate-800 cursor-pointer">
                <input
                  type="checkbox"
                  checked={isSlippery}
                  onChange={(e) => setIsSlippery(e.target.checked)}
                  className="accent-blue-600"
                />
                Stochastic Ice (is_slippery=True)
              </label>
              <label className="flex items-center gap-2 text-slate-600">
                Episodes ({episodes.toLocaleString()})
                <input
                  type="range"
                  min={1000}
                  max={10000}
                  step={500}
                  value={episodes}
                  onChange={(e) => setEpisodes(Number(e.target.value))}
                  className="w-24 accent-blue-600"
                />
              </label>
              <label className="flex items-center gap-2 text-slate-600">
                α ({alpha.toFixed(2)})
                <input
                  type="range"
                  min={0.05}
                  max={0.4}
                  step={0.05}
                  value={alpha}
                  onChange={(e) => setAlpha(Number(e.target.value))}
                  className="w-20 accent-blue-600"
                />
              </label>
              <label className="flex items-center gap-2 text-slate-600">
                γ ({gamma.toFixed(2)})
                <input
                  type="range"
                  min={0.85}
                  max={0.99}
                  step={0.01}
                  value={gamma}
                  onChange={(e) => setGamma(Number(e.target.value))}
                  className="w-20 accent-blue-600"
                />
              </label>
            </div>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricTile
              label="Random Policy Success"
              value={`${(lakeResult.randomSuccessRate * 100).toFixed(1)}%`}
              detail="500-episode baseline"
            />
            <MetricTile
              label="Trained Q-Policy Success"
              value={`${(lakeResult.trainedSuccessRate * 100).toFixed(1)}%`}
              detail={isSlippery ? 'Stochastic 4x4 grid' : 'Deterministic 4x4 grid'}
            />
            <MetricTile
              label="Max State Value"
              value={Math.max(...lakeResult.Q.flat()).toFixed(3)}
              detail="max Q(s, a)"
            />
            <MetricTile
              label="Replay Outcome"
              value={
                lakeSteps[lakeSteps.length - 1]?.reward === 1
                  ? 'Goal Reached'
                  : 'Fell / Timeout'
              }
              detail={`${lakeSteps.length - 1} steps (seed=${rolloutSeed})`}
            />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
            <div className="xl:col-span-5 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-slate-900">
                  Learned Policy Grid & Episode Replay
                </h4>
                <div className="flex items-center gap-2 text-xs">
                  <button
                    type="button"
                    onClick={() => {
                      if (stepIndex >= lakeSteps.length - 1) setStepIndex(0);
                      setPlayingLake((p) => !p);
                    }}
                    className="px-2.5 py-1 bg-blue-600 text-white rounded font-medium hover:bg-blue-700 whitespace-nowrap"
                  >
                    {playingLake ? 'Pause' : 'Play'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setRolloutSeed((s) => s + 1)}
                    className="px-2.5 py-1 border border-slate-200 rounded text-slate-700 hover:bg-slate-50 whitespace-nowrap"
                  >
                    New Seed
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-2 max-w-[320px] mx-auto">
                {FROZEN_LAKE_GRID.flatMap((rowStr, r) =>
                  rowStr.split('').map((tile, c) => {
                    const state = r * 4 + c;
                    const qRow = lakeResult.Q[state];
                    const maxQ = Math.max(...qRow);
                    let bestA = 0;
                    for (let a = 1; a < 4; a++) if (qRow[a] > qRow[bestA]) bestA = a;
                    const isAgentHere =
                      currentLakeFrame?.row === r && currentLakeFrame?.col === c;

                    let bg = 'bg-slate-50 border-slate-200 text-slate-700';
                    if (tile === 'H')
                      bg = 'bg-rose-50 border-rose-200 text-rose-700';
                    else if (tile === 'G')
                      bg = 'bg-emerald-50 border-emerald-200 text-emerald-700';
                    else if (tile === 'S')
                      bg = 'bg-blue-50/70 border-blue-200 text-blue-800';

                    return (
                      <div
                        key={state}
                        className={`relative aspect-square rounded-md border flex flex-col items-center justify-center ${bg} ${
                          isAgentHere ? 'ring-2 ring-blue-600' : ''
                        }`}
                      >
                        <span className="text-[10px] font-mono uppercase opacity-70">
                          {tile}
                        </span>
                        <span className="text-lg font-bold leading-none mt-0.5">
                          {tile === 'H'
                            ? '✕'
                            : tile === 'G'
                            ? '★'
                            : FROZEN_ARROWS[bestA]}
                        </span>
                        <span className="text-[10px] font-mono tabular-nums mt-1 opacity-75">
                          {maxQ.toFixed(2)}
                        </span>
                      </div>
                    );
                  })
                )}
              </div>

              <div className="flex items-center justify-between text-xs font-mono text-slate-600 pt-2 border-t border-slate-100">
                <span>
                  Step {currentLakeFrame?.step ?? 0} / {lakeSteps.length - 1}
                </span>
                <span>
                  Pos ({currentLakeFrame?.row ?? 0}, {currentLakeFrame?.col ?? 0})
                </span>
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, lakeSteps.length - 1)}
                  value={stepIndex}
                  onChange={(e) => {
                    setPlayingLake(false);
                    setStepIndex(Number(e.target.value));
                  }}
                  className="w-24 accent-blue-600"
                />
              </div>
            </div>

            <div className="xl:col-span-7 space-y-6">
              <MultiLineChart
                title="Q-Learning Reward Moving Average"
                xLabel="Episode"
                yLabel="Success Rate"
                yDomain={[0, 1]}
                series={[
                  {
                    name: 'Moving Average Reward',
                    color: '#2563EB',
                    data: lakeResult.movingAvg.map((v, i) => ({ x: i, y: v })),
                  },
                ]}
              />
              <MultiLineChart
                title="Epsilon-Greedy Exploration Decay"
                xLabel="Episode"
                yLabel="Epsilon (ε)"
                yDomain={[0, 1]}
                height={190}
                series={[
                  {
                    name: 'Epsilon',
                    color: '#D97706',
                    data: lakeResult.epsHistory.map((v, i) => ({ x: i, y: v })),
                  },
                ]}
              />
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="bg-white border border-slate-200 rounded-lg p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
            <div className="flex flex-wrap items-center gap-5">
              <label className="flex items-center gap-2 text-slate-600">
                Training Episodes ({dqnEpisodes})
                <input
                  type="range"
                  min={100}
                  max={360}
                  step={20}
                  value={dqnEpisodes}
                  onChange={(e) => setDqnEpisodes(Number(e.target.value))}
                  className="w-28 accent-blue-600"
                />
              </label>
              <button
                type="button"
                onClick={() => setDqnSeed((s) => s + 1)}
                className="px-3 py-1.5 border border-slate-200 rounded-md text-slate-700 hover:bg-slate-50 font-medium"
              >
                Retrain New Seed
              </button>
            </div>
            <span className="text-slate-500 font-mono">
              MLP 4→24→24→2 · Replay Buffer 6,000 · Target Sync 200 steps
            </span>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricTile
              label="Random Policy Avg"
              value={dqnResult.randomAvgReturn.toFixed(1)}
              unit="steps"
              detail="15 eval episodes"
            />
            <MetricTile
              label="Trained DQN Avg"
              value={dqnResult.trainedAvgReturn.toFixed(1)}
              unit="steps"
              detail="15 eval episodes (max 500)"
            />
            <MetricTile
              label="Best Replay Length"
              value={dqnResult.bestEpisodeTrajectory.length.toString()}
              unit="steps"
              detail="Balanced pole duration"
            />
            <MetricTile
              label="Final Epsilon"
              value={dqnResult.epsHistory[
                dqnResult.epsHistory.length - 1
              ].toFixed(3)}
              detail="Exploration rate"
            />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
            <div className="xl:col-span-6 bg-white border border-slate-200 rounded-lg p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-slate-900">
                  Live CartPole-v1 Physics Replay
                </h4>
                <button
                  type="button"
                  onClick={() => setPlayingCart((p) => !p)}
                  className="px-3 py-1 text-xs font-medium bg-blue-600 text-white rounded hover:bg-blue-700 whitespace-nowrap"
                >
                  {playingCart ? 'Pause' : 'Play'}
                </button>
              </div>
              <canvas
                ref={cartCanvasRef}
                width={520}
                height={260}
                className="w-full h-auto rounded-md border border-slate-200"
              />
              {currentCartFrame && (
                <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-100 text-xs font-mono tabular-nums text-slate-700">
                  <div>
                    Step: {currentCartFrame.step + 1} /{' '}
                    {dqnResult.bestEpisodeTrajectory.length}
                  </div>
                  <div>Cart x: {currentCartFrame.state[0].toFixed(3)} m</div>
                  <div>
                    Angle θ:{' '}
                    {((currentCartFrame.state[2] * 180) / Math.PI).toFixed(2)}°
                  </div>
                </div>
              )}
            </div>

            <div className="xl:col-span-6">
              <MultiLineChart
                title="DQN Episode Returns & Moving Average"
                xLabel="Episode"
                yLabel="Episode Return (Steps)"
                series={[
                  {
                    name: 'Episode Return',
                    color: '#93C5FD',
                    data: dqnResult.returnsHistory.map((v, i) => ({
                      x: i,
                      y: v,
                    })),
                  },
                  {
                    name: 'Moving Average (15 ep)',
                    color: '#2563EB',
                    data: dqnResult.movingAvg.map((v, i) => ({
                      x: i + 14,
                      y: v,
                    })),
                  },
                ]}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
