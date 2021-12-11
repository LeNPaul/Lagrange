"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.trackWebVitalMetric = trackWebVitalMetric;
exports.useExperimentalWebVitalsReport = useExperimentalWebVitalsReport;
exports.webVitalsCallbacks = void 0;
var _react = require("react");
const webVitalsCallbacks = new Set();
exports.webVitalsCallbacks = webVitalsCallbacks;
const metrics = [];
function trackWebVitalMetric(metric) {
    metrics.push(metric);
    webVitalsCallbacks.forEach((callback)=>callback(metric)
    );
}
function useExperimentalWebVitalsReport(callback) {
    const metricIndexRef = (0, _react).useRef(0);
    (0, _react).useEffect(()=>{
        // Flush calculated metrics
        const reportMetric = (metric)=>{
            callback(metric);
            metricIndexRef.current = metrics.length;
        };
        for(let i = metricIndexRef.current; i < metrics.length; i++){
            reportMetric(metrics[i]);
        }
        webVitalsCallbacks.add(reportMetric);
        return ()=>{
            webVitalsCallbacks.delete(reportMetric);
        };
    }, [
        callback
    ]);
}

//# sourceMappingURL=vitals.js.map