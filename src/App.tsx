import React, { useState } from 'react';
import { BreastCancerLab } from './components/BreastCancerLab';
import { LinearRegressionLab } from './components/LinearRegressionLab';
import { LogisticRegressionLab } from './components/LogisticRegressionLab';
import { ReinforcementLab } from './components/ReinforcementLab';
import { SvmLab } from './components/SvmLab';
import { TextClassificationLab } from './components/TextClassificationLab';

type LabId =
  | 'linear'
  | 'logistic'
  | 'breast-cancer'
  | 'svm'
  | 'text'
  | 'rl';

interface LabMeta {
  id: LabId;
  title: string;
  domain: string;
  source: string;
}

const LABS: LabMeta[] = [
  {
    id: 'linear',
    title: 'Linear Regression',
    domain: 'Supervised · insurance.csv',
    source: 'LinearRegression.ipynb',
  },
  {
    id: 'logistic',
    title: 'Logistic Regression',
    domain: 'Supervised · heart_disease_data.csv',
    source: 'logistic-heart-disease-regression.ipynb',
  },
  {
    id: 'breast-cancer',
    title: 'Breast Cancer Detection',
    domain: 'Tabular + Vision · Wisconsin CSV',
    source: 'BreastCancerDetection.ipynb',
  },
  {
    id: 'svm',
    title: 'Support Vector Machines',
    domain: 'Kernels + SCOP Homology',
    source: 'SVM_2D & Protein_Homology',
  },
  {
    id: 'text',
    title: 'Text Classification',
    domain: 'NLP · TF-IDF + Logistic',
    source: 'TextClassfication.ipynb',
  },
  {
    id: 'rl',
    title: 'Reinforcement Learning',
    domain: 'Q-Learning & DQN CartPole',
    source: 'FrozenLake & DQN_CartPole',
  },
];

export const App: React.FC = () => {
  const [activeLab, setActiveLab] = useState<LabId>('linear');
  const [resetKey, setResetKey] = useState(0);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      {/* Top Bar Contract: 3 zones separated by gap-8 */}
      <header className="sticky top-0 z-20 bg-white border-b border-slate-200 px-6 py-3.5 flex items-center justify-between gap-8">
        <button
          type="button"
          onClick={() => setActiveLab('linear')}
          className="text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0 text-left"
        >
          ML Workbench
        </button>

        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600">
          <button
            type="button"
            onClick={() => setActiveLab('linear')}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 ${
              activeLab === 'linear' || activeLab === 'logistic'
                ? 'text-slate-900 underline underline-offset-4 decoration-blue-600'
                : ''
            }`}
          >
            Supervised
          </button>
          <button
            type="button"
            onClick={() => setActiveLab('svm')}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 ${
              activeLab === 'svm'
                ? 'text-slate-900 underline underline-offset-4 decoration-blue-600'
                : ''
            }`}
          >
            SVM & Bio
          </button>
          <button
            type="button"
            onClick={() => setActiveLab('breast-cancer')}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 ${
              activeLab === 'breast-cancer' || activeLab === 'text'
                ? 'text-slate-900 underline underline-offset-4 decoration-blue-600'
                : ''
            }`}
          >
            Vision & NLP
          </button>
          <button
            type="button"
            onClick={() => setActiveLab('rl')}
            className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 ${
              activeLab === 'rl'
                ? 'text-slate-900 underline underline-offset-4 decoration-blue-600'
                : ''
            }`}
          >
            Reinforcement
          </button>
        </nav>

        <div className="flex items-center shrink-0">
          <button
            type="button"
            onClick={() => setResetKey((k) => k + 1)}
            className="px-4 py-2 text-xs font-medium text-white bg-slate-900 rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
          >
            Reset Module
          </button>
        </div>
      </header>

      {/* Main Split Console */}
      <div className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-6 flex flex-col lg:flex-row gap-6">
        <aside className="w-full lg:w-72 shrink-0 space-y-4">
          <div className="bg-white border border-slate-200 rounded-lg p-3">
            <div className="px-2 pb-2 mb-2 border-b border-slate-100 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Notebooks & Experiments
            </div>
            <div className="space-y-1">
              {LABS.map((lab, idx) => {
                const selected = lab.id === activeLab;
                return (
                  <button
                    key={lab.id}
                    type="button"
                    onClick={() => setActiveLab(lab.id)}
                    className={`w-full text-left px-3 py-2.5 rounded-md transition-colors ${
                      selected
                        ? 'bg-blue-50 text-blue-950'
                        : 'text-slate-700 hover:bg-slate-50'
                    }`}
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-sm font-semibold">
                        0{idx + 1}. {lab.title}
                      </span>
                    </div>
                    <div className="mt-0.5 text-xs text-slate-500 truncate">
                      {lab.domain}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-4 text-xs text-slate-600 space-y-2 leading-relaxed">
            <div className="font-semibold text-slate-900">
              Runtime Environment
            </div>
            <p>
              All algorithms (OLS, IRLS Logistic Regression, Platt SMO SVM, TF-IDF,
              Tabular Q-Learning, and CartPole DQN) execute directly in TypeScript in
              your browser.
            </p>
            <p className="text-slate-500">
              Real CSVs loaded: <span className="font-mono">insurance.csv</span>,{' '}
              <span className="font-mono">heart_disease_data.csv</span>,{' '}
              <span className="font-mono">breast-cancer-wisconsin-original.csv</span>.
            </p>
          </div>
        </aside>

        <main key={`${activeLab}-${resetKey}`} className="flex-1 min-w-0">
          {activeLab === 'linear' && <LinearRegressionLab />}
          {activeLab === 'logistic' && <LogisticRegressionLab />}
          {activeLab === 'breast-cancer' && <BreastCancerLab />}
          {activeLab === 'svm' && <SvmLab />}
          {activeLab === 'text' && <TextClassificationLab />}
          {activeLab === 'rl' && <ReinforcementLab />}
        </main>
      </div>
    </div>
  );
};

export default App;
