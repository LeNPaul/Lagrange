"use strict";

exports.__esModule = true;
exports.computeId = computeId;
exports.computeSelector = computeSelector;

var _stringHash = _interopRequireDefault(require("string-hash"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

var sanitize = function sanitize(rule) {
  return rule.replace(/\/style/gi, '\\/style');
};

var cache = {};
/**
 * computeId
 *
 * Compute and memoize a jsx id from a basedId and optionally props.
 */

function computeId(baseId, props) {
  if (!props) {
    return "jsx-" + baseId;
  }

  var propsToString = String(props);
  var key = baseId + propsToString;

  if (!cache[key]) {
    cache[key] = "jsx-" + (0, _stringHash["default"])(baseId + "-" + propsToString);
  }

  return cache[key];
}
/**
 * computeSelector
 *
 * Compute and memoize dynamic selectors.
 */


function computeSelector(id, css) {
  var selectoPlaceholderRegexp = /__jsx-style-dynamic-selector/g; // Sanitize SSR-ed CSS.
  // Client side code doesn't need to be sanitized since we use
  // document.createTextNode (dev) and the CSSOM api sheet.insertRule (prod).

  if (typeof window === 'undefined') {
    css = sanitize(css);
  }

  var idcss = id + css;

  if (!cache[idcss]) {
    cache[idcss] = css.replace(selectoPlaceholderRegexp, id);
  }

  return cache[idcss];
}