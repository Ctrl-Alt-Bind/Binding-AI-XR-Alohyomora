 // CStyle_Casting.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>

using namespace std;

class Monster
{
public:
	Monster();
	~Monster();

private:

public:
	int m_hp = 0;
};

Monster::Monster()
{
	cout << "Monster()" << endl;
}

Monster::~Monster()
{
	cout << "~Monster()" << endl;
}
class Player {
public:
	Monster* m_target;
};
Player* FindPlayer(int id) {

	return nullptr;
}

int main()
{
    std::cout << "Hello World!\n";

	// malloc, free -> 생성자 호출x.
	void* ptr = malloc(1000); // 1000 바이트 만큼 할당

	Monster* m = (Monster*)ptr;
	m->m_hp = 100;

	free(ptr);

	// new, delete -> 생성자 호출함.
	Monster* m1 = new Monster();
	m1->m_hp = 200;
	delete m1; // delete 안 해주면 메모리 누수나는 상황

	// 1) 크래시? 2)그냥 된다? -> 실행은 됨.
	// 분명히 메모리 해제하고 날렸는데? 
	// Use-After-Free 상황을 어떻게 예방할 수 있는가?
	m1 = nullptr; // -> 0의 주소값을 가짐.

	m1->m_hp = 300;

	Player* player = FindPlayer(0);
	if (player != nullptr) {
		// 예외 처리 코드를 만들어 대응
	}

}

// 프로그램 실행: <Ctrl+F5> 또는 [디버그] > [디버깅하지 않고 시작] 메뉴
// 프로그램 디버그: <F5> 키 또는 [디버그] > [디버깅 시작] 메뉴

// 시작을 위한 팁: 
//   1. [솔루션 탐색기] 창을 사용하여 파일을 추가/관리합니다.
//   2. [팀 탐색기] 창을 사용하여 소스 제어에 연결합니다.
//   3. [출력] 창을 사용하여 빌드 출력 및 기타 메시지를 확인합니다.
//   4. [오류 목록] 창을 사용하여 오류를 봅니다.
//   5. [프로젝트] > [새 항목 추가]로 이동하여 새 코드 파일을 만들거나, [프로젝트] > [기존 항목 추가]로 이동하여 기존 코드 파일을 프로젝트에 추가합니다.
//   6. 나중에 이 프로젝트를 다시 열려면 [파일] > [열기] > [프로젝트]로 이동하고 .sln 파일을 선택합니다.
