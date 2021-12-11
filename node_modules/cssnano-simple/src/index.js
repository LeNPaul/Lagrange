const createSimplePreset = require('cssnano-preset-simple');

module.exports = (opts = {}, postcss = require('postcss')) => {
  const excludeAll = Boolean(opts && opts.excludeAll);

  const userOpts = Object.assign({}, opts);
  if (excludeAll) {
    for (const userOption in userOpts) {
      if (!userOpts.hasOwnProperty(userOption)) continue;
      const val = userOpts[userOption];
      if (!Boolean(val)) {
        continue;
      }

      if (Object.prototype.toString.call(val) === '[object Object]') {
        userOpts[userOption] = Object.assign({}, { exclude: false }, val);
      }
    }
  }

  const options = Object.assign(
    {},
    excludeAll ? { rawCache: true } : undefined,
    userOpts
  );

  const plugins = [];
  createSimplePreset(options).plugins.forEach((plugin) => {
    if (Array.isArray(plugin)) {
      const [processor, opts] = plugin;

      const isEnabled =
        // No options:
        (!excludeAll && typeof opts === 'undefined') ||
        // Short-hand enabled:
        (typeof opts === 'boolean' && opts) ||
        // Include all plugins:
        (!excludeAll && opts && typeof opts === 'object' && !opts.exclude) ||
        // Exclude all plugins:
        (excludeAll &&
          opts &&
          typeof opts === 'object' &&
          opts.exclude === false);

      if (isEnabled) {
        plugins.push(processor(opts));
      }
    } else {
      plugins.push(plugin);
    }
  });

  return postcss(plugins);
};

module.exports.postcss = true;
