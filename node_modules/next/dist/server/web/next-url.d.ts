import type { DomainLocale, I18NConfig } from '../config-shared';
/**
 * TODO
 *
 * - Add comments to the URLNext API.
 * - Move internals to be using symbols for its shape.
 * - Make sure logging does not show any implementation details.
 * - Include in the event payload the nextJS configuration
 */
interface Options {
    basePath?: string;
    headers?: {
        [key: string]: string | string[] | undefined;
    };
    i18n?: I18NConfig | null;
    trailingSlash?: boolean;
}
export declare class NextURL extends URL {
    private _basePath;
    private _locale?;
    private _options;
    private _url;
    constructor(input: string, options?: Options);
    get absolute(): boolean;
    analyzeUrl(): void;
    formatPathname(): string;
    get locale(): string;
    set locale(locale: string);
    get defaultLocale(): string | undefined;
    get domainLocale(): DomainLocale | undefined;
    get searchParams(): URLSearchParams;
    get host(): string;
    set host(value: string);
    get hostname(): string;
    set hostname(value: string);
    get port(): string;
    set port(value: string);
    get protocol(): string;
    set protocol(value: string);
    get href(): string;
    set href(url: string);
    get origin(): string;
    get pathname(): string;
    set pathname(value: string);
    get hash(): string;
    set hash(value: string);
    get search(): string;
    set search(value: string);
    get password(): string;
    set password(value: string);
    get username(): string;
    set username(value: string);
    get basePath(): string;
    set basePath(value: string);
    toString(): string;
    toJSON(): string;
}
export {};
