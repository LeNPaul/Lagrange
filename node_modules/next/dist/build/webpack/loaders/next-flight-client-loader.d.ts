/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
declare type Source = string | ArrayBuffer | Uint8Array;
export default function transformSource(this: any, source: Source): Promise<Source>;
export {};
