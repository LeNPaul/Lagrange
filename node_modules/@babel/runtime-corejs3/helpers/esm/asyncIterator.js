import _Symbol from "@babel/runtime-corejs3/core-js/symbol";
import _Symbol$asyncIterator from "@babel/runtime-corejs3/core-js/symbol/async-iterator";
import _Symbol$iterator from "@babel/runtime-corejs3/core-js/symbol/iterator";
import _Promise from "@babel/runtime-corejs3/core-js/promise";
export default function _asyncIterator(iterable) {
  var method,
      async,
      sync,
      retry = 2;

  if (typeof _Symbol !== "undefined") {
    async = _Symbol$asyncIterator;
    sync = _Symbol$iterator;
  }

  while (retry--) {
    if (async && (method = iterable[async]) != null) {
      return method.call(iterable);
    }

    if (sync && (method = iterable[sync]) != null) {
      return new AsyncFromSyncIterator(method.call(iterable));
    }

    async = "@@asyncIterator";
    sync = "@@iterator";
  }

  throw new TypeError("Object is not async iterable");
}

function AsyncFromSyncIterator(s) {
  AsyncFromSyncIterator = function AsyncFromSyncIterator(s) {
    this.s = s;
    this.n = s.next;
  };

  AsyncFromSyncIterator.prototype = {
    s: null,
    n: null,
    next: function next() {
      return AsyncFromSyncIteratorContinuation(this.n.apply(this.s, arguments));
    },
    "return": function _return(value) {
      var ret = this.s["return"];

      if (ret === undefined) {
        return _Promise.resolve({
          value: value,
          done: true
        });
      }

      return AsyncFromSyncIteratorContinuation(ret.apply(this.s, arguments));
    },
    "throw": function _throw(value) {
      var thr = this.s["return"];
      if (thr === undefined) return _Promise.reject(value);
      return AsyncFromSyncIteratorContinuation(thr.apply(this.s, arguments));
    }
  };

  function AsyncFromSyncIteratorContinuation(r) {
    if (Object(r) !== r) {
      return _Promise.reject(new TypeError(r + " is not an object."));
    }

    var done = r.done;
    return _Promise.resolve(r.value).then(function (value) {
      return {
        value: value,
        done: done
      };
    });
  }

  return new AsyncFromSyncIterator(s);
}