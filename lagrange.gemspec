# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = "Fatik's Blog"
  spec.version       = "3.0.7"
  spec.authors       = ["Fatik Owais"]
  spec.email         = ["fatikowais@gmail.com"]

  spec.summary       = "A personal blog to share stuff"
  spec.homepage      = "https://github.com/fatik/fatik.github.io"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select { |f| f.match(%r!^(assets|_layouts|_includes|_sass|LICENSE|README)!i) }

  spec.add_runtime_dependency "jekyll", "~> 3.7.4"

  spec.add_runtime_dependency "jekyll-paginate", "~> 1.1"
  spec.add_runtime_dependency "jekyll-sitemap", "~> 1.2"
  spec.add_runtime_dependency "jekyll-seo-tag", "~> 2.5.0"

  spec.add_development_dependency "bundler", "~> 1.16"
  spec.add_development_dependency "rake", "~> 12.0"
end
