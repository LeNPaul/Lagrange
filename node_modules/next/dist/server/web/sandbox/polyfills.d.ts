import { Crypto as WebCrypto } from 'next/dist/compiled/@peculiar/webcrypto';
import { CryptoKey } from 'next/dist/compiled/@peculiar/webcrypto';
export declare function atob(b64Encoded: string): string;
export declare function btoa(str: string): string;
export { CryptoKey };
export declare class Crypto extends WebCrypto {
    randomUUID: any;
}
export declare class ReadableStream<T> {
    constructor(opts?: UnderlyingSource);
}
