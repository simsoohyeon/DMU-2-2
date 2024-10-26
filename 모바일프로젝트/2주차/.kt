package com.example.m02_layoutview // 패키지 선언, 해당 파일이 속한 패키지 이름을 정의

import android.os.Bundle // Bundle 클래스를 import하여 안드로이드의 activity 생명주기에서 필요한 데이터 저장 및 복원 기능을 사용할 수 있도록 함
import androidx.appcompat.app.AppCompatActivity // AppCompatActivity를 import, 안드로이드 앱의 기본 액티비티 클래스
import com.example.m02_layoutview.databinding.ActivityMainBinding // View Binding을 위한 ActivityMainBinding 클래스 import

class MainActivity : AppCompatActivity() { // AppCompatActivity를 상속받아 MainActivity 클래스 정의
    override fun onCreate(savedInstanceState: Bundle?) { // Activity가 생성될 때 호출되는 메소드
        super.onCreate(savedInstanceState) // 부모 클래스의 onCreate 메소드 호출, 기본 초기화 수행

        val binding = ActivityMainBinding.inflate(layoutInflater) // View Binding을 통해 layoutInflater를 사용해 XML 레이아웃을 바인딩

        setContentView(binding.root) // 바인딩된 뷰를 현재 액티비티의 콘텐츠 뷰로 설정

        binding.textTitle.setText(getString(R.string.title)) // 문자열 리소스를 사용하여 textTitle에 제목 설정
        // setContentView(R.layout.activity_main) // 이전 방식으로 레이아웃을 설정하는 코드 (주석처리된 부분)

        // val textView = findViewById<TextView>(R.id.textTitle) // findViewById를 사용하여 TextView 참조 (주석처리됨)
        // textView.text = getString(R.string.title) // 주석처리된 findViewById 방식으로 제목 설정

        // textView.setText("동양나눔장터") // 특정 텍스트를 설정하는 주석처리된 코드
    }
}
