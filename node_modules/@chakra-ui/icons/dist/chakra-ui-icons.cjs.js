'use strict';

if (process.env.NODE_ENV === "production") {
  module.exports = require("./chakra-ui-icons.cjs.prod.js");
} else {
  module.exports = require("./chakra-ui-icons.cjs.dev.js");
}
