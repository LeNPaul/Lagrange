---
published: true
comments: true
layout: post
title: String Hashing in C++
author: Riad Afridi Shibly
categories: programming
tags: [c++, hasing]
image: 2019-04-22-string-hashing-in-cpp/banner.jpg
---

## What is Hashing?

Let's think about a function `unsigned int hash(string)`. This function will take a string as a parameter and return a unique ID let's say an unsigned integer. This is called a hash function. It'll generate a unique ID for every unique string. And same unique ID over and over for same string. Things get complicated when the two different strings generate same ID. This is called a hash collision.

## Hashing in C++
Let's use `std::hash` to generate hash for some string.

```cpp
#include <functional> // std::hash
#include <iostream>

int main() {
    std::hash<std::string> str_hash;
    std::cout << str_hash("Hello") << '\n';
    std::cout << str_hash("World") << '\n';
}
```

This syntax can be somewhat simplified to this. (maybe ?)

```cpp
#include <functional> // std::hash
#include <iostream>

int main() {
    std::cout << std::hash<std::string>{}("Hello World!") << '\n';
}
```

Well, we've used the `std::hash` function to generate hash. But we don't know yet which algorithm it uses! Let's use some easy string hashing algorithm.

## [djb2 Hashing algorithm](http://www.cse.yorku.ca/~oz/hash.html#djb2)

This algorithm is very simple yet powerful enough to generate some good hash.

```cpp
unsigned long hash(unsigned char *str) {
    unsigned long hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}
```

Let's use this algorithm.

```cpp
#include <iostream>

namespace djb2 {
std::size_t hash(uint8_t *str) {
    std::size_t hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

std::size_t hash(const std::string &str) {
    return hash((uint8_t *)str.c_str());
}

} // namespace djb2

int main() {
    std::string s = "Hello World";
    std::cout << djb2::hash(s) << '\n';
}
```
