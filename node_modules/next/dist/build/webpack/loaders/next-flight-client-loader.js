"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.default = transformSource;
var acorn = _interopRequireWildcard(require("acorn"));
function _interopRequireWildcard(obj) {
    if (obj && obj.__esModule) {
        return obj;
    } else {
        var newObj = {
        };
        if (obj != null) {
            for(var key in obj){
                if (Object.prototype.hasOwnProperty.call(obj, key)) {
                    var desc = Object.defineProperty && Object.getOwnPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : {
                    };
                    if (desc.get || desc.set) {
                        Object.defineProperty(newObj, key, desc);
                    } else {
                        newObj[key] = obj[key];
                    }
                }
            }
        }
        newObj.default = obj;
        return newObj;
    }
}
let stashedResolve = null;
function addExportNames(names, node) {
    switch(node.type){
        case 'Identifier':
            names.push(node.name);
            return;
        case 'ObjectPattern':
            for(let i = 0; i < node.properties.length; i++)addExportNames(names, node.properties[i]);
            return;
        case 'ArrayPattern':
            for(let i1 = 0; i1 < node.elements.length; i1++){
                const element = node.elements[i1];
                if (element) addExportNames(names, element);
            }
            return;
        case 'Property':
            addExportNames(names, node.value);
            return;
        case 'AssignmentPattern':
            addExportNames(names, node.left);
            return;
        case 'RestElement':
            addExportNames(names, node.argument);
            return;
        case 'ParenthesizedExpression':
            addExportNames(names, node.expression);
            return;
        default:
            return;
    }
}
function resolveClientImport(specifier, parentURL) {
    // Resolve an import specifier as if it was loaded by the client. This doesn't use
    // the overrides that this loader does but instead reverts to the default.
    // This resolution algorithm will not necessarily have the same configuration
    // as the actual client loader. It should mostly work and if it doesn't you can
    // always convert to explicit exported names instead.
    const conditions = [
        'node',
        'import'
    ];
    if (stashedResolve === null) {
        throw new Error('Expected resolve to have been called before transformSource');
    }
    return stashedResolve(specifier, {
        conditions,
        parentURL
    }, stashedResolve);
}
async function parseExportNamesInto(transformedSource, names, parentURL, loadModule) {
    const { body  } = acorn.parse(transformedSource, {
        ecmaVersion: 11,
        sourceType: 'module'
    });
    for(let i = 0; i < body.length; i++){
        const node = body[i];
        switch(node.type){
            case 'ExportAllDeclaration':
                if (node.exported) {
                    addExportNames(names, node.exported);
                    continue;
                } else {
                    const { url  } = await resolveClientImport(node.source.value, parentURL);
                    const source = '';
                    parseExportNamesInto(source, names, url, loadModule);
                    continue;
                }
            case 'ExportDefaultDeclaration':
                names.push('default');
                continue;
            case 'ExportNamedDeclaration':
                if (node.declaration) {
                    if (node.declaration.type === 'VariableDeclaration') {
                        const declarations = node.declaration.declarations;
                        for(let j = 0; j < declarations.length; j++){
                            addExportNames(names, declarations[j].id);
                        }
                    } else {
                        addExportNames(names, node.declaration.id);
                    }
                }
                if (node.specificers) {
                    const specificers = node.specificers;
                    for(let j = 0; j < specificers.length; j++){
                        addExportNames(names, specificers[j].exported);
                    }
                }
                continue;
            default:
                break;
        }
    }
}
async function transformSource(source) {
    const { resourcePath , resourceQuery  } = this;
    if (resourceQuery !== '?flight') return source;
    let url = resourcePath;
    const transformedSource = source;
    if (typeof transformedSource !== 'string') {
        throw new Error('Expected source to have been transformed to a string.');
    }
    const names = [];
    await parseExportNamesInto(transformedSource, names, url + resourceQuery, this.loadModule);
    // next.js/packages/next/<component>.js
    if (/[\\/]next[\\/](link|image)\.js$/.test(url)) {
        names.push('default');
    }
    let newSrc = "const MODULE_REFERENCE = Symbol.for('react.module.reference');\n";
    for(let i = 0; i < names.length; i++){
        const name = names[i];
        if (name === 'default') {
            newSrc += 'export default ';
        } else {
            newSrc += 'export const ' + name + ' = ';
        }
        newSrc += '{ $$typeof: MODULE_REFERENCE, filepath: ';
        newSrc += JSON.stringify(url);
        newSrc += ', name: ';
        newSrc += JSON.stringify(name);
        newSrc += '};\n';
    }
    return newSrc;
}

//# sourceMappingURL=next-flight-client-loader.js.map