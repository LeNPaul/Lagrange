# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = "jekyll-theme-harveynick-lagrange"
  spec.version       = "2.1.0"
  spec.authors       = ["Nick Johnson"]
  spec.email         = ["contact@harveynick.com"]

  spec.summary       = "A minimalist Jekyll theme for running a personal blog. Fork of Lagrange by Paul Le https://github.com/LeNPaul/Lagrange."
  spec.homepage      = "https://github.com/harveynick/lagrange"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select { |f| f.match(%r!^(assets|_layouts|_includes|_sass|LICENSE|README)!i) }

  spec.add_runtime_dependency "jekyll", "~> 3.7.4"

  spec.add_development_dependency "bundler", "~> 1.16"
  spec.add_development_dependency "rake", "~> 12.0"
end
