import { fitLogisticRegression, type ExecutionMetrics } from './mlMath';
import { trainTestSplitIndices } from './csv';

export interface ReviewSample {
  id: number;
  label: 0 | 1; // 0 = negative, 1 = positive
  text: string;
}

export const SAMPLE_YELP_REVIEWS: ReviewSample[] = [
  { id: 1, label: 1, text: 'Absolutely fantastic pasta and attentive service. The truffle risotto was unforgettable and worth every penny.' },
  { id: 2, label: 1, text: 'Best espresso in town! Cozy atmosphere, warm croissants straight from the oven, and friendly baristas.' },
  { id: 3, label: 1, text: 'Our server made great recommendations and the wood-fired pizza crust was crisp and smoky. Will definitely return.' },
  { id: 4, label: 1, text: 'Fresh sushi, generous portions, and super clean dining room. The spicy tuna roll and miso soup were outstanding.' },
  { id: 5, label: 1, text: 'Quick lunch stop that exceeded expectations. Crispy falafel, bright lemon dressing, and super polite staff.' },
  { id: 6, label: 1, text: 'Celebrated our anniversary here and everything was immaculate. Tender steak, balanced cocktails, and zero wait.' },
  { id: 7, label: 1, text: 'Hidden gem with authentic tacos and vibrant homemade salsa. Great value and fast kitchen turnaround.' },
  { id: 8, label: 1, text: 'Warm hospitality from the moment we walked in. Loved the seasonal salad and flaky sea salt chocolate tart.' },
  { id: 9, label: 1, text: 'Consistently delicious Thai curries with fresh herbs and balanced spice. Takeout always arrives piping hot.' },
  { id: 10, label: 1, text: 'Impressive craft beer list and juicy smash burgers with crispy fries. Fun patio vibe on weekend evenings.' },
  { id: 11, label: 1, text: 'The ramen broth had deep umami flavor and the chashu melted in my mouth. Best comfort meal around.' },
  { id: 12, label: 1, text: 'Spotless bakery with flaky pastries and rich almond croissants. Worth the morning line every single time.' },
  { id: 13, label: 1, text: 'Brilliant brunch spot! Fluffy pancakes, crisp bacon, and bottomless coffee refilled before we even asked.' },
  { id: 14, label: 1, text: 'Very accommodating with food allergies and every dish tasted vibrant and fresh. Highly recommend the grilled salmon.' },
  { id: 15, label: 1, text: 'Wonderful patio seating, upbeat music, and phenomenal wood-roasted chicken with garlic potatoes.' },
  { id: 16, label: 1, text: 'Clean, bright, and welcoming. The pho broth was fragrant and packed with fresh basil and tender beef.' },
  { id: 17, label: 1, text: 'Great neighborhood bistro with thoughtful wine pairings and creamy burrata appetizer. Truly memorable dinner.' },
  { id: 18, label: 1, text: 'Fast, reliable, and delicious dumplings every visit. Crispy bottoms and savory filling.' },
  { id: 19, label: 1, text: 'Superbly cooked seafood paella with crisp socarrat and sweet shrimp. Staff checked on us just the right amount.' },
  { id: 20, label: 1, text: 'Amazing gelato flavors that taste like real fruit and roasted pistachio. Friendly scoopers and generous servings.' },
  { id: 21, label: 1, text: 'Top-notch barbecue brisket with tender bark and tangy house sauce. Mac and cheese side was creamy perfection.' },
  { id: 22, label: 1, text: 'Everything we ordered was flavorful and beautifully plated. Loved the roasted cauliflower and tahini glaze.' },
  { id: 23, label: 1, text: 'Smooth check-in, spotless tables, and incredible butter chicken with warm garlic naan.' },
  { id: 24, label: 1, text: 'Refreshing iced matcha and delicious avocado toast on thick sourdough. My new favorite study cafe.' },
  { id: 25, label: 1, text: 'Friendly owner who greets regulars by name. The Reuben sandwich is stacked high and grilled to golden crispness.' },
  { id: 26, label: 1, text: 'Delicious vegetarian options that feel creative and satisfying. The mushroom flatbread stole the show.' },
  { id: 27, label: 1, text: 'Perfectly seared scallops and attentive sommelier. Pricey but truly special occasion quality.' },
  { id: 28, label: 1, text: 'Great happy hour oysters and crisp fries. Bartender crafted an awesome old fashioned.' },
  { id: 29, label: 1, text: 'Always fresh ingredients, generous bowls, and super fast online pickup counter.' },
  { id: 30, label: 1, text: 'Charming interior, comforting homemade lasagna, and decadent tiramisu to finish.' },
  { id: 31, label: 1, text: 'Fantastic dim sum cart service with hot har gow and crispy pork buns.' },
  { id: 32, label: 1, text: 'Crispy Korean fried chicken with sweet-spicy glaze that stays crunchy even after delivery.' },
  { id: 33, label: 1, text: 'Loved the spicy margaritas and smoky charred street corn. Service was cheerful and quick.' },
  { id: 34, label: 1, text: 'Outstanding poke bowl with fresh ahi tuna, mango, and crunchy garlic crisps.' },
  { id: 35, label: 1, text: 'Reliable neighborhood favorite. Great prices, huge portions, and consistently cheerful staff.' },
  { id: 36, label: 1, text: 'The lamb shank fell right off the bone and the saffron rice was fragrant and fluffy.' },
  { id: 37, label: 0, text: 'Waited forty-five minutes after our reservation time only to receive lukewarm pasta and zero apology.' },
  { id: 38, label: 0, text: 'Terrible service. Our waiter disappeared for thirty minutes and forgot two of our drinks completely.' },
  { id: 39, label: 0, text: 'Overpriced and bland. The burger was dry, the fries were soggy, and the bun fell apart immediately.' },
  { id: 40, label: 0, text: 'Awful experience from start to finish. Sticky tables, loud clattering kitchen, and cold soup.' },
  { id: 41, label: 0, text: 'Wrong takeout order again and missing half the sauces. Nobody answered the phone when we called back.' },
  { id: 42, label: 0, text: 'Fish smelled questionable and the rice tasted stale. Definitely will not be coming back here.' },
  { id: 43, label: 0, text: 'Tiny portions for twenty-eight dollars and everything tasted overly salty and greasy.' },
  { id: 44, label: 0, text: 'Rude host rolled his eyes when we asked for a clean booth. Food took forever and arrived burnt.' },
  { id: 45, label: 0, text: 'Watery iced coffee and stale muffins that felt three days old. Total waste of money.' },
  { id: 46, label: 0, text: 'Pizza crust was limp and soaked in grease, and the toppings slid right off the slice.' },
  { id: 47, label: 0, text: 'Slow kitchen and disorganized staff. We waited an hour for basic sandwiches on a quiet Tuesday.' },
  { id: 48, label: 0, text: 'Disappointing ramen with lukewarm broth and rubbery noodles. Not worth half the price.' },
  { id: 49, label: 0, text: 'Manager was dismissive when we pointed out the raw chicken in the middle of our entree.' },
  { id: 50, label: 0, text: 'Way too noisy to hold a conversation and the air conditioning was broken. Left before dessert.' },
  { id: 51, label: 0, text: 'Soggy nachos, canned cheese taste, and watered-down drinks. Avoid this tourist trap.' },
  { id: 52, label: 0, text: 'Steak ordered medium-rare came out well-done and chewy like leather. No check-in from server.' },
  { id: 53, label: 0, text: 'Dirty silverware and smudged water glasses right at the table. Completely unacceptable hygiene.' },
  { id: 54, label: 0, text: 'Delivery arrived ninety minutes late and completely cold. Packaging leaked all over the bag.' },
  { id: 55, label: 0, text: 'Flavorless curry that tasted like plain tomato paste with tough, overcooked chicken.' },
  { id: 56, label: 0, text: 'Charged an unexpected service fee and refused to fix an obvious billing mistake on the receipt.' },
  { id: 57, label: 0, text: 'Dry pancakes, burnt hash browns, and rude cashier. Plenty of better breakfast spots nearby.' },
  { id: 58, label: 0, text: 'Sushi rice was mushy and warm, and the avocado inside was brown. Very disappointing.' },
  { id: 59, label: 0, text: 'Zero seasoning on the vegetables and the pasta was mushy and overboiled.' },
  { id: 60, label: 0, text: 'Hostess seated walk-ins ahead of us despite our confirmed booking. Never returning.' },
  { id: 61, label: 0, text: 'Greasy fried rice and rubbery shrimp. Left most of the plate untouched.' },
  { id: 62, label: 0, text: 'Coffee tasted burnt and bitter, and the pastry case had flies near the croissants.' },
  { id: 63, label: 0, text: 'Understaffed and chaotic. Took twenty minutes just to get menus and water.' },
  { id: 64, label: 0, text: 'Tacos were cold, tortillas tore instantly, and the salsa tasted straight out of a jar.' },
  { id: 65, label: 0, text: 'Really wanted to like it, but the chicken was dry and the salad greens were wilted.' },
  { id: 66, label: 0, text: 'Bad value for money. Skimpy toppings, burnt edges, and indifferent staff.' },
  { id: 67, label: 0, text: 'Unpleasant chemical smell near the entrance and lukewarm tap water.' },
  { id: 68, label: 0, text: 'Forgot our appetizer entirely and still tried to keep it on the final bill.' },
  { id: 69, label: 0, text: 'Bland soup, stale bread, and slow checkout. Save your time and eat elsewhere.' },
  { id: 70, label: 0, text: 'Overcooked salmon that was chalky and dry. Sides were cold when served.' },
  { id: 71, label: 0, text: 'Terrible acoustics, cramped seating, and mediocre food that felt microwaved.' },
  { id: 72, label: 0, text: 'Waited forever for the check after asking twice. Food was forgettable at best.' },
];

const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at',
  'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
  'during', 'before', 'after', 'to', 'from', 'in', 'out', 'on', 'off', 'over',
  'under', 'again', 'further', 'here', 'there', 'all', 'any', 'both', 'each',
  'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
  'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', 'now',
  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
  'do', 'does', 'did', 'of', 'it', 'its', 'this', 'that', 'these', 'those',
  'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'his', 'she', 'her',
  'they', 'their', 'them', 'as', 'from', 'what', 'which', 'who', 'whom',
]);

export function tokenizeNgrams(text: string): string[] {
  const words = text
    .toLowerCase()
    .replace(/[^a-z0-9\s']/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 1 && !STOP_WORDS.has(w));
  const tokens = [...words];
  for (let i = 0; i < words.length - 1; i++) {
    tokens.push(`${words[i]} ${words[i + 1]}`);
  }
  return tokens;
}

export interface TrainedTextClassifier {
  vocab: string[];
  idf: number[];
  weights: number[];
  intercept: number;
  trainSamples: ReviewSample[];
  testSamples: ReviewSample[];
  testProbs: number[];
  testPreds: number[];
  execution: ExecutionMetrics;
  predictOne: (text: string) => {
    label: 0 | 1;
    probability: number;
    topTokens: { token: string; weight: number }[];
  };
}

export function trainTfidfLogistic(
  samples: ReviewSample[],
  options: { testSize?: number; seed?: number; C?: number } = {}
): TrainedTextClassifier {
  const t0 = performance.now();
  const testSize = options.testSize ?? 0.25;
  const seed = options.seed ?? 42;
  const C = options.C ?? 2.0;

  const { trainIdx, testIdx } = trainTestSplitIndices(
    samples.length,
    testSize,
    seed,
    samples.map((s) => s.label)
  );

  const trainSamples = trainIdx.map((i) => samples[i]);
  const testSamples = testIdx.map((i) => samples[i]);

  const docFreq = new Map<string, number>();
  const trainTokens = trainSamples.map((s) => {
    const toks = tokenizeNgrams(s.text);
    const seen = new Set(toks);
    for (const t of seen) {
      docFreq.set(t, (docFreq.get(t) ?? 0) + 1);
    }
    return toks;
  });

  // Keep terms appearing in >=1 doc (higher for large corpora, >=1 for 72 samples)
  const vocab = Array.from(docFreq.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 450)
    .map(([t]) => t);

  const vocabIdx = new Map(vocab.map((t, i) => [t, i]));
  const N = trainSamples.length;
  const idf = vocab.map((t) => Math.log((1 + N) / (1 + (docFreq.get(t) ?? 0))) + 1);

  const vectorize = (toks: string[]): number[] => {
    const vec = Array(vocab.length).fill(0);
    const tf = new Map<number, number>();
    for (const t of toks) {
      const idx = vocabIdx.get(t);
      if (idx !== undefined) tf.set(idx, (tf.get(idx) ?? 0) + 1);
    }
    let sqSum = 0;
    for (const [idx, count] of tf.entries()) {
      const val = count * idf[idx];
      vec[idx] = val;
      sqSum += val * val;
    }
    const norm = Math.sqrt(sqSum) || 1;
    for (let i = 0; i < vec.length; i++) vec[i] /= norm;
    return vec;
  };

  const XTrain = trainTokens.map(vectorize);
  const yTrain = trainSamples.map((s) => s.label);
  const XTest = testSamples.map((s) => vectorize(tokenizeNgrams(s.text)));

  const model = fitLogisticRegression(XTrain, yTrain, { C, maxIter: 500 });
  const testProbs = model.predictProba(XTest);
  const testPreds = testProbs.map((p) => (p >= 0.5 ? 1 : 0));
  const totalTimeMs = performance.now() - t0;

  const predictOne = (text: string) => {
    const toks = tokenizeNgrams(text);
    const vec = vectorize(toks);
    const prob = model.predictProbaOne(vec);
    const contribs: { token: string; weight: number }[] = [];
    const seen = new Set<string>();
    for (const t of toks) {
      if (seen.has(t)) continue;
      seen.add(t);
      const idx = vocabIdx.get(t);
      if (idx !== undefined) {
        contribs.push({ token: t, weight: vec[idx] * model.weights[idx] });
      }
    }
    contribs.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
    return {
      label: (prob >= 0.5 ? 1 : 0) as 0 | 1,
      probability: prob,
      topTokens: contribs.slice(0, 6),
    };
  };

  return {
    vocab,
    idf,
    weights: model.weights,
    intercept: model.intercept,
    trainSamples,
    testSamples,
    testProbs,
    testPreds,
    execution: {
      ...model.execution,
      trainingTimeMs: totalTimeMs,
      solver: `TF-IDF + ${model.execution.solver}`,
    },
    predictOne,
  };
}
