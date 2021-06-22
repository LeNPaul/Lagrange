---
layout: post
title:  "Getting stuck with stuff like a pro"
categories: [Outreachy]
image: inexplicable_2x.png
author : Noor
---


##### Outreachy Entry 2 & Entry 3 (fused)

This is my 2nd entry during the Outreachy Internship, It's July 2020, I'm almost done with the internship and I've been falling behind on writing entries, but better late than never. That's why I'll be doing a long entry where I fused up two, I will try to explain what part of the Firefox for iOS app I am handling and document as much as possible the challenges that faced me while working on that part of the app.


# Treading In  

### The Why? 

While speaking about our project we were told to try and explain it in layman's terms to newcomers in the community, it was an exercise. While chatting with the outreachy community on the topic, I thought that yes, understanding of things definetly deepens and how inclusive you are to others contributing to it can be managed by using simpler terms instead of just jargons.
then someone suggested these websites which are really cool :

[Hemingway App](http://hemingwayapp.com)

[The Up-Goer Five text editor](https://splasho.com/upgoer5/) this one is inspired by an [XKCD Comic](https://xkcd.com/1133/)

It was a hard challenge for me, since I'm more used to the technical terms, and sometimes you can't explain something technical because simply there's no words to accomodate for it in normal terms.


### The What?

I will try as an exercise to use these tools and explain what I'm working on in the Firefox iOS app. I might not stick to all the rules of the exercise and fail but I'll try to make my words as simple as possible.

I'm working on the browser's Widget. A browser: is the program you use to navigate the internet and go to websites and it's called an 'app' on mobile phones. A widget is : an extra part of the app which can be accessed from your home screen to do some actions without opening the app itself. it's not an app, it's just a menu for quick actions if I may say. When I joined the firefox team, the widget had two features (think of a feature as a benefit for the user or a utility) : Opening New broswer tabs in normal mode, opening a tab in private mode and navigating to a copied link in the user's clipboard. The design was old and needed redesigning as well.

I was required to redesign the widget and add more features to it. The process went as follows : 

1. Doing research on what features might be useful to the user to have there or were requested in user reviews. There was an opened spike to come up with useful features to add to the widget and the UX team sharing the redesign with the engineering team. 
    
    [Spike](https://github.com/mozilla-mobile/firefox-ios/issues/6661#issuecomment-64577666) 

2. After redesigning the app, came up issues that need to be handled after testing with the new design. After migrating to the new visual design, a code refactoring was in need. We wanted to shift the widget's code to the MVVM architecture.

3. After making a design document for the refactoring and refactoring the code to a new architecture, came the time for speculating the features like how to handle the navigating to the copied link better and also adding another feature which was closing all private browsing tabs from the widget. 

    [Go to copied link feature opened issue](https://github.com/mozilla-mobile/firefox-ios/issues/6935)

    [Go to copied link feature PR](https://github.com/mozilla-mobile/firefox-ios/pull/6956)

    [Close private tabs feature opened issue](https://github.com/mozilla-mobile/firefox-ios/issues/6794)

    [Close private tabs feature PR](https://github.com/mozilla-mobile/firefox-ios/pull/6971#pullrequestreview-458879860)

4. On WWDC 2020, Apple introduced entirely new widgets to iOS, with a new API and they were on the homescreen instead of residing in the notification center. That introduced new opportunities to explore for me since I was working on the widget and a call for having a home screen widget for firefox à la iOS 14 was much expected.

5. I had to do research on how Apple's Widget Kit works, so I checked their WWDC talks and code alongs and made a summarized doc for the API. Luckily, a contributer made a PR for the new design using Swift UI, which I took and built on top of it the new features that we were building for the old Today Widget.

    [UX research for the new iOS14 Widgets](https://github.com/mozilla-mobile/firefox-ios/issues/6936#event-3576519888)

    [iOS14 Widget Implementation](https://github.com/mozilla-mobile/firefox-ios/pull/7051)

6. I'm at milestone 6 now! At the moment, new designs and features for iOS14 are being researched and I'm working on polishing the code + writing Unit tests for the widget.

You can find all the conversations made regarding both features and how they were implemented in both the issues and PRs.


### The How? 
###### Nobody said it was eaaasaaey

In this section, as you might have guessed, I was going to speak about the struggles. because there is no How without a "WHY GOD AM I GETTING THIS ERROR! FOR GOD'S SAKE!"

I will try to categorize the problems I faced and how they were overcome (as much as my memory allows me to) some of which were merely because it was my first time handling these kind of issue and some were just because my laptop or Xcode were just haunted 🤷🏻‍

1. Snapkit and Constraints

2. Buttons

3. Stackviews

4. the MVVM

5. Accessibility

6. Carthage issues

7. Working with multiple Xcodes/Swift versions and command line tools

8. support for earlier iOS versions

9. The Today Widget as an independent target


### The state of mind while untangling the misery of the How

### Tips from your Ma











