/// <reference types="node" />
import { IncomingMessage, ServerResponse } from 'http';
import { UrlWithParsedQuery } from 'url';
import { NextConfig } from './config-shared';
import Server from './next-server';
export declare function imageOptimizer(server: Server, req: IncomingMessage, res: ServerResponse, parsedUrl: UrlWithParsedQuery, nextConfig: NextConfig, distDir: string, isDev?: boolean): Promise<{
    finished: boolean;
}>;
/**
 * Inspects the first few bytes of a buffer to determine if
 * it matches the "magic number" of known file signatures.
 * https://en.wikipedia.org/wiki/List_of_file_signatures
 */
export declare function detectContentType(buffer: Buffer): "image/svg+xml" | "image/avif" | "image/webp" | "image/png" | "image/jpeg" | "image/gif" | null;
export declare function getMaxAge(str: string | null): number;
export declare function resizeImage(content: Buffer, dimension: 'width' | 'height', size: number, extension: 'avif' | 'webp' | 'png' | 'jpeg', quality: number): Promise<Buffer>;
export declare function getImageSize(buffer: Buffer, extension: 'avif' | 'webp' | 'png' | 'jpeg'): Promise<{
    width?: number;
    height?: number;
}>;
