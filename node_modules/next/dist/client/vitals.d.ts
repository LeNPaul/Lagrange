import { NextWebVitalsMetric } from '../pages/_app';
declare type ReportWebVitalsCallback = (webVitals: NextWebVitalsMetric) => any;
export declare const webVitalsCallbacks: Set<ReportWebVitalsCallback>;
export declare function trackWebVitalMetric(metric: NextWebVitalsMetric): void;
export declare function useExperimentalWebVitalsReport(callback: ReportWebVitalsCallback): void;
export {};
