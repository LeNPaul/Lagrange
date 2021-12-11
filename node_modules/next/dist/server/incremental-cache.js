"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
var _fs = require("fs");
var _lruCache = _interopRequireDefault(require("next/dist/compiled/lru-cache"));
var _path = _interopRequireDefault(require("path"));
var _constants = require("../shared/lib/constants");
var _normalizePagePath = require("./normalize-page-path");
function _interopRequireDefault(obj) {
    return obj && obj.__esModule ? obj : {
        default: obj
    };
}
function toRoute(pathname) {
    return pathname.replace(/\/$/, '').replace(/\/index$/, '') || '/';
}
class IncrementalCache {
    constructor({ max , dev , distDir , pagesDir , flushToDisk , locales  }){
        this.incrementalOptions = {
            dev,
            distDir,
            pagesDir,
            flushToDisk: !dev && (typeof flushToDisk !== 'undefined' ? flushToDisk : true)
        };
        this.locales = locales;
        if (dev) {
            this.prerenderManifest = {
                version: -1,
                routes: {
                },
                dynamicRoutes: {
                },
                notFoundRoutes: [],
                preview: null
            };
        } else {
            this.prerenderManifest = JSON.parse((0, _fs).readFileSync(_path.default.join(distDir, _constants.PRERENDER_MANIFEST), 'utf8'));
        }
        if (process.env.__NEXT_TEST_MAX_ISR_CACHE) {
            // Allow cache size to be overridden for testing purposes
            max = parseInt(process.env.__NEXT_TEST_MAX_ISR_CACHE, 10);
        }
        if (max) {
            this.cache = new _lruCache.default({
                max,
                length ({ value  }) {
                    if (!value || value.kind === 'REDIRECT') return 25;
                    // rough estimate of size of cache value
                    return value.html.length + JSON.stringify(value.pageData).length;
                }
            });
        }
    }
    getSeedPath(pathname, ext) {
        return _path.default.join(this.incrementalOptions.pagesDir, `${pathname}.${ext}`);
    }
    calculateRevalidate(pathname, fromTime) {
        pathname = toRoute(pathname);
        // in development we don't have a prerender-manifest
        // and default to always revalidating to allow easier debugging
        if (this.incrementalOptions.dev) return new Date().getTime() - 1000;
        const { initialRevalidateSeconds  } = this.prerenderManifest.routes[pathname] || {
            initialRevalidateSeconds: 1
        };
        const revalidateAfter = typeof initialRevalidateSeconds === 'number' ? initialRevalidateSeconds * 1000 + fromTime : initialRevalidateSeconds;
        return revalidateAfter;
    }
    getFallback(page) {
        page = (0, _normalizePagePath).normalizePagePath(page);
        return _fs.promises.readFile(this.getSeedPath(page, 'html'), 'utf8');
    }
    // get data from cache if available
    async get(pathname) {
        if (this.incrementalOptions.dev) return null;
        pathname = (0, _normalizePagePath).normalizePagePath(pathname);
        let data = this.cache && this.cache.get(pathname);
        // let's check the disk for seed data
        if (!data) {
            if (this.prerenderManifest.notFoundRoutes.includes(pathname)) {
                const now = Date.now();
                const revalidateAfter = this.calculateRevalidate(pathname, now);
                data = {
                    value: null,
                    revalidateAfter: revalidateAfter !== false ? now : false
                };
            }
            try {
                const htmlPath = this.getSeedPath(pathname, 'html');
                const html = await _fs.promises.readFile(htmlPath, 'utf8');
                const { mtime  } = await _fs.promises.stat(htmlPath);
                const pageData = JSON.parse(await _fs.promises.readFile(this.getSeedPath(pathname, 'json'), 'utf8'));
                data = {
                    revalidateAfter: this.calculateRevalidate(pathname, mtime.getTime()),
                    value: {
                        kind: 'PAGE',
                        html,
                        pageData
                    }
                };
                if (this.cache) {
                    this.cache.set(pathname, data);
                }
            } catch (_) {
            // unable to get data from disk
            }
        }
        if (!data) {
            return null;
        }
        if (data && data.revalidateAfter !== false && data.revalidateAfter < new Date().getTime()) {
            data.isStale = true;
        }
        const manifestPath = toRoute(pathname);
        const manifestEntry = this.prerenderManifest.routes[manifestPath];
        if (data && manifestEntry) {
            data.curRevalidate = manifestEntry.initialRevalidateSeconds;
        }
        return data;
    }
    // populate the incremental cache with new data
    async set(pathname, data, revalidateSeconds) {
        if (this.incrementalOptions.dev) return;
        if (typeof revalidateSeconds !== 'undefined') {
            // TODO: Update this to not mutate the manifest from the
            // build.
            this.prerenderManifest.routes[pathname] = {
                dataRoute: _path.default.posix.join('/_next/data', `${(0, _normalizePagePath).normalizePagePath(pathname)}.json`),
                srcRoute: null,
                initialRevalidateSeconds: revalidateSeconds
            };
        }
        pathname = (0, _normalizePagePath).normalizePagePath(pathname);
        if (this.cache) {
            this.cache.set(pathname, {
                revalidateAfter: this.calculateRevalidate(pathname, new Date().getTime()),
                value: data
            });
        }
        // TODO: This option needs to cease to exist unless it stops mutating the
        // `next build` output's manifest.
        if (this.incrementalOptions.flushToDisk && (data === null || data === void 0 ? void 0 : data.kind) === 'PAGE') {
            try {
                const seedPath = this.getSeedPath(pathname, 'html');
                await _fs.promises.mkdir(_path.default.dirname(seedPath), {
                    recursive: true
                });
                await _fs.promises.writeFile(seedPath, data.html, 'utf8');
                await _fs.promises.writeFile(this.getSeedPath(pathname, 'json'), JSON.stringify(data.pageData), 'utf8');
            } catch (error) {
                // failed to flush to disk
                console.warn('Failed to update prerender files for', pathname, error);
            }
        }
    }
}
exports.IncrementalCache = IncrementalCache;

//# sourceMappingURL=incremental-cache.js.map