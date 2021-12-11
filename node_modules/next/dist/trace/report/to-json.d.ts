declare const localEndpoint: {
    serviceName: string;
    ipv4: string;
    port: number;
};
declare type Event = {
    traceId: string;
    parentId?: string;
    name: string;
    id: string;
    timestamp: number;
    duration: number;
    localEndpoint?: typeof localEndpoint;
    tags?: Object;
};
export declare function batcher(reportEvents: (evts: Event[]) => Promise<void>): {
    flushAll: () => Promise<void>;
    report: (event: Event) => void;
};
declare const _default: {
    flushAll: () => Promise<void> | undefined;
    report: (name: string, duration: number, timestamp: number, id: string, parentId?: string | undefined, attrs?: Object | undefined) => void;
};
export default _default;
