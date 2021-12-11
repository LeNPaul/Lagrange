"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.imageOptimizer = imageOptimizer;
exports.detectContentType = detectContentType;
exports.getMaxAge = getMaxAge;
exports.resizeImage = resizeImage;
exports.getImageSize = getImageSize;
var _accept = require("@hapi/accept");
var _crypto = require("crypto");
var _fs = require("fs");
var _getOrientation = require("get-orientation");
var _imageSize = _interopRequireDefault(require("image-size"));
var _isAnimated = _interopRequireDefault(require("next/dist/compiled/is-animated"));
var _contentDisposition = _interopRequireDefault(require("next/dist/compiled/content-disposition"));
var _path = require("path");
var _stream = _interopRequireDefault(require("stream"));
var _url = _interopRequireDefault(require("url"));
var _fileExists = require("../lib/file-exists");
var _imageConfig = require("./image-config");
var _main = require("./lib/squoosh/main");
var _sendPayload = require("./send-payload");
var _serveStatic = require("./serve-static");
var _chalk = _interopRequireDefault(require("chalk"));
function _interopRequireDefault(obj) {
    return obj && obj.__esModule ? obj : {
        default: obj
    };
}
const AVIF = 'image/avif';
const WEBP = 'image/webp';
const PNG = 'image/png';
const JPEG = 'image/jpeg';
const GIF = 'image/gif';
const SVG = 'image/svg+xml';
const CACHE_VERSION = 3;
const ANIMATABLE_TYPES = [
    WEBP,
    PNG,
    GIF
];
const VECTOR_TYPES = [
    SVG
];
const BLUR_IMG_SIZE = 8 // should match `next-image-loader`
;
const inflightRequests = new Map();
let sharp;
try {
    sharp = require(process.env.NEXT_SHARP_PATH || 'sharp');
} catch (e) {
// Sharp not present on the server, Squoosh fallback will be used
}
let showSharpMissingWarning = process.env.NODE_ENV === 'production';
async function imageOptimizer(server, req, res, parsedUrl, nextConfig, distDir, isDev = false) {
    const imageData = nextConfig.images || _imageConfig.imageConfigDefault;
    const { deviceSizes =[] , imageSizes =[] , domains =[] , loader , minimumCacheTTL =60 , formats =[
        'image/webp'
    ] ,  } = imageData;
    if (loader !== 'default') {
        await server.render404(req, res, parsedUrl);
        return {
            finished: true
        };
    }
    const { headers  } = req;
    const { url , w , q  } = parsedUrl.query;
    const mimeType = getSupportedMimeType(formats, headers.accept);
    let href;
    if (!url) {
        res.statusCode = 400;
        res.end('"url" parameter is required');
        return {
            finished: true
        };
    } else if (Array.isArray(url)) {
        res.statusCode = 400;
        res.end('"url" parameter cannot be an array');
        return {
            finished: true
        };
    }
    let isAbsolute;
    if (url.startsWith('/')) {
        href = url;
        isAbsolute = false;
    } else {
        let hrefParsed;
        try {
            hrefParsed = new URL(url);
            href = hrefParsed.toString();
            isAbsolute = true;
        } catch (_error) {
            res.statusCode = 400;
            res.end('"url" parameter is invalid');
            return {
                finished: true
            };
        }
        if (![
            'http:',
            'https:'
        ].includes(hrefParsed.protocol)) {
            res.statusCode = 400;
            res.end('"url" parameter is invalid');
            return {
                finished: true
            };
        }
        if (!domains.includes(hrefParsed.hostname)) {
            res.statusCode = 400;
            res.end('"url" parameter is not allowed');
            return {
                finished: true
            };
        }
    }
    if (!w) {
        res.statusCode = 400;
        res.end('"w" parameter (width) is required');
        return {
            finished: true
        };
    } else if (Array.isArray(w)) {
        res.statusCode = 400;
        res.end('"w" parameter (width) cannot be an array');
        return {
            finished: true
        };
    }
    if (!q) {
        res.statusCode = 400;
        res.end('"q" parameter (quality) is required');
        return {
            finished: true
        };
    } else if (Array.isArray(q)) {
        res.statusCode = 400;
        res.end('"q" parameter (quality) cannot be an array');
        return {
            finished: true
        };
    }
    // Should match output from next-image-loader
    const isStatic = url.startsWith(`${nextConfig.basePath || ''}/_next/static/media`);
    const width = parseInt(w, 10);
    if (!width || isNaN(width)) {
        res.statusCode = 400;
        res.end('"w" parameter (width) must be a number greater than 0');
        return {
            finished: true
        };
    }
    const sizes = [
        ...deviceSizes,
        ...imageSizes
    ];
    if (isDev) {
        sizes.push(BLUR_IMG_SIZE);
    }
    if (!sizes.includes(width)) {
        res.statusCode = 400;
        res.end(`"w" parameter (width) of ${width} is not allowed`);
        return {
            finished: true
        };
    }
    const quality = parseInt(q);
    if (isNaN(quality) || quality < 1 || quality > 100) {
        res.statusCode = 400;
        res.end('"q" parameter (quality) must be a number between 1 and 100');
        return {
            finished: true
        };
    }
    const hash = getHash([
        CACHE_VERSION,
        href,
        width,
        quality,
        mimeType
    ]);
    const imagesDir = (0, _path).join(distDir, 'cache', 'images');
    const hashDir = (0, _path).join(imagesDir, hash);
    const now = Date.now();
    // If there're concurrent requests hitting the same resource and it's still
    // being optimized, wait before accessing the cache.
    if (inflightRequests.has(hash)) {
        await inflightRequests.get(hash);
    }
    let dedupeResolver;
    inflightRequests.set(hash, new Promise((resolve)=>dedupeResolver = resolve
    ));
    try {
        if (await (0, _fileExists).fileExists(hashDir, 'directory')) {
            const files = await _fs.promises.readdir(hashDir);
            for (let file of files){
                const [maxAgeStr, expireAtSt, etag, extension] = file.split('.');
                const maxAge = Number(maxAgeStr);
                const expireAt = Number(expireAtSt);
                const contentType = (0, _serveStatic).getContentType(extension);
                const fsPath = (0, _path).join(hashDir, file);
                if (now < expireAt) {
                    const result = setResponseHeaders(req, res, url, etag, maxAge, contentType, isStatic, isDev);
                    if (!result.finished) {
                        (0, _fs).createReadStream(fsPath).pipe(res);
                    }
                    return {
                        finished: true
                    };
                } else {
                    await _fs.promises.unlink(fsPath);
                }
            }
        }
        let upstreamBuffer;
        let upstreamType;
        let maxAge;
        if (isAbsolute) {
            const upstreamRes = await fetch(href);
            if (!upstreamRes.ok) {
                res.statusCode = upstreamRes.status;
                res.end('"url" parameter is valid but upstream response is invalid');
                return {
                    finished: true
                };
            }
            res.statusCode = upstreamRes.status;
            upstreamBuffer = Buffer.from(await upstreamRes.arrayBuffer());
            upstreamType = detectContentType(upstreamBuffer) || upstreamRes.headers.get('Content-Type');
            maxAge = getMaxAge(upstreamRes.headers.get('Cache-Control'));
        } else {
            try {
                const resBuffers = [];
                const mockRes = new _stream.default.Writable();
                const isStreamFinished = new Promise(function(resolve, reject) {
                    mockRes.on('finish', ()=>resolve(true)
                    );
                    mockRes.on('end', ()=>resolve(true)
                    );
                    mockRes.on('error', ()=>reject()
                    );
                });
                mockRes.write = (chunk)=>{
                    resBuffers.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
                };
                mockRes._write = (chunk)=>{
                    mockRes.write(chunk);
                };
                const mockHeaders = {
                };
                mockRes.writeHead = (_status, _headers)=>Object.assign(mockHeaders, _headers)
                ;
                mockRes.getHeader = (name)=>mockHeaders[name.toLowerCase()]
                ;
                mockRes.getHeaders = ()=>mockHeaders
                ;
                mockRes.getHeaderNames = ()=>Object.keys(mockHeaders)
                ;
                mockRes.setHeader = (name, value)=>mockHeaders[name.toLowerCase()] = value
                ;
                mockRes.removeHeader = (name)=>{
                    delete mockHeaders[name.toLowerCase()];
                };
                mockRes._implicitHeader = ()=>{
                };
                mockRes.connection = res.connection;
                mockRes.finished = false;
                mockRes.statusCode = 200;
                const mockReq = new _stream.default.Readable();
                mockReq._read = ()=>{
                    mockReq.emit('end');
                    mockReq.emit('close');
                    return Buffer.from('');
                };
                mockReq.headers = req.headers;
                mockReq.method = req.method;
                mockReq.url = href;
                mockReq.connection = req.connection;
                await server.getRequestHandler()(mockReq, mockRes, _url.default.parse(href, true));
                await isStreamFinished;
                res.statusCode = mockRes.statusCode;
                upstreamBuffer = Buffer.concat(resBuffers);
                upstreamType = detectContentType(upstreamBuffer) || mockRes.getHeader('Content-Type');
                maxAge = getMaxAge(mockRes.getHeader('Cache-Control'));
            } catch (err) {
                res.statusCode = 500;
                res.end('"url" parameter is valid but upstream response is invalid');
                return {
                    finished: true
                };
            }
        }
        const expireAt = Math.max(maxAge, minimumCacheTTL) * 1000 + now;
        if (upstreamType) {
            const vector = VECTOR_TYPES.includes(upstreamType);
            const animate = ANIMATABLE_TYPES.includes(upstreamType) && (0, _isAnimated).default(upstreamBuffer);
            if (vector || animate) {
                await writeToCacheDir(hashDir, upstreamType, maxAge, expireAt, upstreamBuffer);
                sendResponse(req, res, url, maxAge, upstreamType, upstreamBuffer, isStatic, isDev);
                return {
                    finished: true
                };
            }
            if (!upstreamType.startsWith('image/')) {
                res.statusCode = 400;
                res.end("The requested resource isn't a valid image.");
                return {
                    finished: true
                };
            }
        }
        let contentType;
        if (mimeType) {
            contentType = mimeType;
        } else if ((upstreamType === null || upstreamType === void 0 ? void 0 : upstreamType.startsWith('image/')) && (0, _serveStatic).getExtension(upstreamType)) {
            contentType = upstreamType;
        } else {
            contentType = JPEG;
        }
        try {
            let optimizedBuffer;
            if (sharp) {
                // Begin sharp transformation logic
                const transformer = sharp(upstreamBuffer);
                transformer.rotate();
                const { width: metaWidth  } = await transformer.metadata();
                if (metaWidth && metaWidth > width) {
                    transformer.resize(width);
                }
                if (contentType === AVIF) {
                    if (transformer.avif) {
                        const avifQuality = quality - 15;
                        transformer.avif({
                            quality: Math.max(avifQuality, 0),
                            chromaSubsampling: '4:2:0'
                        });
                    } else {
                        console.warn(_chalk.default.yellow.bold('Warning: ') + `Your installed version of the 'sharp' package does not support AVIF images. Run 'yarn add sharp@latest' to upgrade to the latest version.\n` + 'Read more: https://nextjs.org/docs/messages/sharp-version-avif');
                        transformer.webp({
                            quality
                        });
                    }
                } else if (contentType === WEBP) {
                    transformer.webp({
                        quality
                    });
                } else if (contentType === PNG) {
                    transformer.png({
                        quality
                    });
                } else if (contentType === JPEG) {
                    transformer.jpeg({
                        quality
                    });
                }
                optimizedBuffer = await transformer.toBuffer();
            // End sharp transformation logic
            } else {
                // Show sharp warning in production once
                if (showSharpMissingWarning) {
                    console.warn(_chalk.default.yellow.bold('Warning: ') + `For production Image Optimization with Next.js, the optional 'sharp' package is strongly recommended. Run 'yarn add sharp', and Next.js will use it automatically for Image Optimization.\n` + 'Read more: https://nextjs.org/docs/messages/sharp-missing-in-production');
                    showSharpMissingWarning = false;
                }
                // Begin Squoosh transformation logic
                const orientation = await (0, _getOrientation).getOrientation(upstreamBuffer);
                const operations = [];
                if (orientation === _getOrientation.Orientation.RIGHT_TOP) {
                    operations.push({
                        type: 'rotate',
                        numRotations: 1
                    });
                } else if (orientation === _getOrientation.Orientation.BOTTOM_RIGHT) {
                    operations.push({
                        type: 'rotate',
                        numRotations: 2
                    });
                } else if (orientation === _getOrientation.Orientation.LEFT_BOTTOM) {
                    operations.push({
                        type: 'rotate',
                        numRotations: 3
                    });
                } else {
                // TODO: support more orientations
                // eslint-disable-next-line @typescript-eslint/no-unused-vars
                // const _: never = orientation
                }
                operations.push({
                    type: 'resize',
                    width
                });
                if (contentType === AVIF) {
                    optimizedBuffer = await (0, _main).processBuffer(upstreamBuffer, operations, 'avif', quality);
                } else if (contentType === WEBP) {
                    optimizedBuffer = await (0, _main).processBuffer(upstreamBuffer, operations, 'webp', quality);
                } else if (contentType === PNG) {
                    optimizedBuffer = await (0, _main).processBuffer(upstreamBuffer, operations, 'png', quality);
                } else if (contentType === JPEG) {
                    optimizedBuffer = await (0, _main).processBuffer(upstreamBuffer, operations, 'jpeg', quality);
                }
            // End Squoosh transformation logic
            }
            if (optimizedBuffer) {
                await writeToCacheDir(hashDir, contentType, maxAge, expireAt, optimizedBuffer);
                sendResponse(req, res, url, maxAge, contentType, optimizedBuffer, isStatic, isDev);
            } else {
                throw new Error('Unable to optimize buffer');
            }
        } catch (error) {
            sendResponse(req, res, url, maxAge, upstreamType, upstreamBuffer, isStatic, isDev);
        }
        return {
            finished: true
        };
    } finally{
        // Make sure to remove the hash in the end.
        dedupeResolver();
        inflightRequests.delete(hash);
    }
}
async function writeToCacheDir(dir, contentType, maxAge, expireAt, buffer) {
    await _fs.promises.mkdir(dir, {
        recursive: true
    });
    const extension = (0, _serveStatic).getExtension(contentType);
    const etag = getHash([
        buffer
    ]);
    const filename = (0, _path).join(dir, `${maxAge}.${expireAt}.${etag}.${extension}`);
    await _fs.promises.writeFile(filename, buffer);
}
function getFileNameWithExtension(url, contentType) {
    const [urlWithoutQueryParams] = url.split('?');
    const fileNameWithExtension = urlWithoutQueryParams.split('/').pop();
    if (!contentType || !fileNameWithExtension) {
        return;
    }
    const [fileName] = fileNameWithExtension.split('.');
    const extension = (0, _serveStatic).getExtension(contentType);
    return `${fileName}.${extension}`;
}
function setResponseHeaders(req, res, url, etag, maxAge, contentType, isStatic, isDev) {
    res.setHeader('Vary', 'Accept');
    res.setHeader('Cache-Control', isStatic ? 'public, max-age=315360000, immutable' : `public, max-age=${isDev ? 0 : maxAge}, must-revalidate`);
    if ((0, _sendPayload).sendEtagResponse(req, res, etag)) {
        // already called res.end() so we're finished
        return {
            finished: true
        };
    }
    if (contentType) {
        res.setHeader('Content-Type', contentType);
    }
    const fileName = getFileNameWithExtension(url, contentType);
    if (fileName) {
        res.setHeader('Content-Disposition', (0, _contentDisposition).default(fileName, {
            type: 'inline'
        }));
    }
    res.setHeader('Content-Security-Policy', `script-src 'none'; sandbox;`);
    return {
        finished: false
    };
}
function sendResponse(req, res, url, maxAge, contentType, buffer, isStatic, isDev) {
    const etag = getHash([
        buffer
    ]);
    const result = setResponseHeaders(req, res, url, etag, maxAge, contentType, isStatic, isDev);
    if (!result.finished) {
        res.end(buffer);
    }
}
function getSupportedMimeType(options, accept = '') {
    const mimeType = (0, _accept).mediaType(accept, options);
    return accept.includes(mimeType) ? mimeType : '';
}
function getHash(items) {
    const hash = (0, _crypto).createHash('sha256');
    for (let item of items){
        if (typeof item === 'number') hash.update(String(item));
        else {
            hash.update(item);
        }
    }
    // See https://en.wikipedia.org/wiki/Base64#Filenames
    return hash.digest('base64').replace(/\//g, '-');
}
function parseCacheControl(str) {
    const map = new Map();
    if (!str) {
        return map;
    }
    for (let directive of str.split(',')){
        let [key, value] = directive.trim().split('=');
        key = key.toLowerCase();
        if (value) {
            value = value.toLowerCase();
        }
        map.set(key, value);
    }
    return map;
}
function detectContentType(buffer) {
    if ([
        255,
        216,
        255
    ].every((b, i)=>buffer[i] === b
    )) {
        return JPEG;
    }
    if ([
        137,
        80,
        78,
        71,
        13,
        10,
        26,
        10
    ].every((b, i)=>buffer[i] === b
    )) {
        return PNG;
    }
    if ([
        71,
        73,
        70,
        56
    ].every((b, i)=>buffer[i] === b
    )) {
        return GIF;
    }
    if ([
        82,
        73,
        70,
        70,
        0,
        0,
        0,
        0,
        87,
        69,
        66,
        80
    ].every((b, i)=>!b || buffer[i] === b
    )) {
        return WEBP;
    }
    if ([
        60,
        63,
        120,
        109,
        108
    ].every((b, i)=>buffer[i] === b
    )) {
        return SVG;
    }
    if ([
        0,
        0,
        0,
        0,
        102,
        116,
        121,
        112,
        97,
        118,
        105,
        102
    ].every((b, i)=>!b || buffer[i] === b
    )) {
        return AVIF;
    }
    return null;
}
function getMaxAge(str) {
    const map = parseCacheControl(str);
    if (map) {
        let age = map.get('s-maxage') || map.get('max-age') || '';
        if (age.startsWith('"') && age.endsWith('"')) {
            age = age.slice(1, -1);
        }
        const n = parseInt(age, 10);
        if (!isNaN(n)) {
            return n;
        }
    }
    return 0;
}
async function resizeImage(content, dimension, size, // Should match VALID_BLUR_EXT
extension, quality) {
    if (sharp) {
        const transformer = sharp(content);
        if (extension === 'avif') {
            if (transformer.avif) {
                transformer.avif({
                    quality
                });
            } else {
                console.warn(_chalk.default.yellow.bold('Warning: ') + `Your installed version of the 'sharp' package does not support AVIF images. Run 'yarn add sharp@latest' to upgrade to the latest version.\n` + 'Read more: https://nextjs.org/docs/messages/sharp-version-avif');
                transformer.webp({
                    quality
                });
            }
        } else if (extension === 'webp') {
            transformer.webp({
                quality
            });
        } else if (extension === 'png') {
            transformer.png({
                quality
            });
        } else if (extension === 'jpeg') {
            transformer.jpeg({
                quality
            });
        }
        if (dimension === 'width') {
            transformer.resize(size);
        } else {
            transformer.resize(null, size);
        }
        const buf = await transformer.toBuffer();
        return buf;
    } else {
        const resizeOperationOpts = dimension === 'width' ? {
            type: 'resize',
            width: size
        } : {
            type: 'resize',
            height: size
        };
        const buf = await (0, _main).processBuffer(content, [
            resizeOperationOpts
        ], extension, quality);
        return buf;
    }
}
async function getImageSize(buffer, // Should match VALID_BLUR_EXT
extension) {
    // TODO: upgrade "image-size" package to support AVIF
    // See https://github.com/image-size/image-size/issues/348
    if (extension === 'avif') {
        if (sharp) {
            const transformer = sharp(buffer);
            const { width , height  } = await transformer.metadata();
            return {
                width,
                height
            };
        } else {
            const { width , height  } = await (0, _main).decodeBuffer(buffer);
            return {
                width,
                height
            };
        }
    }
    const { width , height  } = (0, _imageSize).default(buffer);
    return {
        width,
        height
    };
}

//# sourceMappingURL=image-optimizer.js.map