FROM ruby:3.1.0-buster

COPY ./Gemfile /tmp/
COPY ./lagrange.gemspec /tmp/
WORKDIR /tmp
RUN bundle install

WORKDIR /pvphan.github.io

RUN git clone https://github.com/mathjax/MathJax.git /tmp/MathJax -b 3.2.0
RUN cp -r /tmp/MathJax/es5 /pvphan.github.io/mathjax
