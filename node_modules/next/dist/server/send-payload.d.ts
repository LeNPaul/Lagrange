/// <reference types="node" />
import { IncomingMessage, ServerResponse } from 'http';
import RenderResult from './render-result';
export declare type PayloadOptions = {
    private: true;
} | {
    private: boolean;
    stateful: true;
} | {
    private: boolean;
    stateful: false;
    revalidate: number | false;
};
export declare function setRevalidateHeaders(res: ServerResponse, options: PayloadOptions): void;
export declare function sendRenderResult({ req, res, result, type, generateEtags, poweredByHeader, options, }: {
    req: IncomingMessage;
    res: ServerResponse;
    result: RenderResult;
    type: 'html' | 'json';
    generateEtags: boolean;
    poweredByHeader: boolean;
    options?: PayloadOptions;
}): Promise<void>;
export declare function sendEtagResponse(req: IncomingMessage, res: ServerResponse, etag: string | undefined): boolean;
