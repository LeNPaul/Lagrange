---
layout: post
title: "아두이노 IDE 설치하기"
author: "Yeonah Ki"
categories: journal
tags: [arduino]
image: download_arduino_ide.png
---



아두이노를 사용하기 위해서 가장먼저 해야할 일은 "아두이노 IDE"를 설치하는 것이다.

아두이노는 "아두이노 IDE"에서 작성한 소스코드를 업로드하여 사용하므로 이 프로그램 없이는 아두이노를 동작하게 할 수 없다.        

<br>

# IDE란?

IDE란 Integrated Development Environment의 약자로 통합 개발 환경을 말한다.

하나의 프로그램 위에서 코딩, 디버깅, 컴파일, 배포(업로드)가 모두 가능함을 의미한다.  

  <br>




# 아두이노 IDE 다운로드 받기

[아두이노 공식 사이트]( www.arduino.cc/en/Main/Software) 에서 무료로 아두이노 IDE를 다운로드 받을 수 있다.

윈도우, 맥, 리눅스까지 다양한 운영체제에서 사용할 수 있도록 지원하고 있으므로 사용하는 운영체제제 맞춰 다운로드 받고 설치하여 사용한다.



![download_arduino_ide](https://user-images.githubusercontent.com/21331671/38652266-24020abc-3e40-11e8-95e8-42281c2a054d.png)







사용하는 운영체제를 클릭하면, "Contribute to the Arduino Software"라는 페이지가 나오며 이 페이지에서 실제 "아두이노 IDE" 프로그램을 다운로드 받는다. 해당 메뉴에서는 아두이노의 활성화를 위해 기부도 함께 받고 있으니, 아두이노 재단에 기부를 할 사람은 "CONTRIBUTE & DOWNLOAD"를, 그냥 다운로드만 받을 사람은 "JUST DOWNLOAD" 를 누른다.  



![contribute_arduino](https://user-images.githubusercontent.com/21331671/38652277-324922d6-3e40-11e8-8d62-abf759992f9d.png)



  

  <br>

## Windows(윈도우)에서 아두이노 IDE 설치하기

아두이노 IDE 를 다운로드 한 후, 윈도우에서 설치하는 방법은 아래 그림과 같다.



![arduino_install_windows](https://user-images.githubusercontent.com/21331671/38652320-6eed9802-3e40-11e8-976b-105f68b7373e.png)





  <br>

## Mac(맥)에서 아두이노 IDE 설치하기

아두이노 IDE 를 다운로드 한 후, 맥에서 설치하는 방법은 아래 그림과 같다.

나의 경우는 다운로드 받은 .zip 파일을 더블클릭으로 압축해제가 가능한 [반디집](https://www.bandisoft.co.kr/bandizip/)이라는 프로그램을 쓰고 있어서 좀 더 쉽게 설치가 가능하였다. 



![arduino_install_mac](https://user-images.githubusercontent.com/21331671/38652387-b6b46206-3e40-11e8-8c97-1ec87f61d1b1.png)



<br><br>

# 아두이노 호환보드를 사용한다면?

정품 아두이노 보드를 사용한다면, 아두이노 IDE  설치 후 바로 사용이 가능하다.

하지만, 흔히 짭두이노라고 불리는 아두이노 호환보드를 사용한다면 추가로 드라이버 설치가 필요할 수 있다.

물론, 드라이버 무설치 버전의 아두이노 호환보드도 있으니 본인이 구매한 보드가 드라이버 설치가 추가로 필요한지 먼저 확인이 필요하다.

가장 많이 사용하는 드라이버는 CH340/CH342SER 드라이버로 [사이트](http://www.wch.cn/download/CH341SER_ZIP.html)에서 다운로드가 가능하다.

해당 드라이버도 사용하는 운영체제에 따라 설치하는 버전이 다르므로 미리 확인이 필요하다.

* 윈도우
  * 압축 해제 후, 디렉토리에서 CH341SER 디렉토리를 선택한다.(INSTALL 디렉토리 아님!!!!)
  * CH341SER 디렉토리 내의 SETUP 파일을 반드시 관리자 권한으로 실행한다.
  * 실행파일에서 INSTALL 버튼을 눌러 설치를 진행하고, 만약 실패할 경우 UNINSTALL  후 다시 INSTALL한다.
* 맥
  * 압축 해제 후, 디렉토리에서 CH34x_Install_V1.4.pkg 파일을 선택한다.
  * '계속' 버튼을 클릭하여 설치를 진행한다.
  * 설치가 완료되면 '재시동'을 클릭하여 PC를 재부팅한다.



