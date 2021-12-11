"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.shouldUseYarn = shouldUseYarn;
var _childProcess = require("child_process");
function shouldUseYarn() {
    try {
        const userAgent = process.env.npm_config_user_agent;
        if (userAgent) {
            return Boolean(userAgent && userAgent.startsWith('yarn'));
        }
        (0, _childProcess).execSync('yarnpkg --version', {
            stdio: 'ignore'
        });
        return true;
    } catch (e) {
        return false;
    }
}

//# sourceMappingURL=should-use-yarn.js.map