/// <reference types="node" />
import type { ServerResponse } from 'http';
import type { Writable } from 'stream';
export declare type NodeWritablePiper = (res: Writable, next: (err?: Error) => void) => void;
export default class RenderResult {
    _result: string | NodeWritablePiper;
    constructor(response: string | NodeWritablePiper);
    toUnchunkedString(): string;
    pipe(res: ServerResponse): Promise<void>;
    isDynamic(): boolean;
    static fromStatic(value: string): RenderResult;
    static empty: RenderResult;
}
