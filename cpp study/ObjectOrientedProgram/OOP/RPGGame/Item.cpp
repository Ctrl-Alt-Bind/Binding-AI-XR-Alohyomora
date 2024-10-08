#include <iostream>
#include "Item.h"

using namespace std;
// ============================
//			Item
// ============================

Item::Item(ItemType itemType) : m_type(itemType) // 자식 클래스의 생성자에서 지정해줘야 함.
{
	int randValue = rand() % 100;

	if (randValue < 50) {
		m_rarity = IR_Normal;
	}
	else if (randValue < 80)
	{
		m_rarity = IR_Rare;
	}
	else
	{
		m_rarity = IR_Unique;
	}
}

Item::~Item()
{
}
void Item::PrintInfo()
{
	switch (m_rarity)
	{
	case IR_Normal:
		cout << "[희귀도] 일반" << endl;
		break;
	case IR_Rare:
		cout << "[희귀도] 레어" << endl;
		break;
	case IR_Unique:
		cout << "[희귀도] 유니크" << endl;
		break;
	default:
		break;
	}
}
// ============================
//			 Weapon
// ============================
Weapon::Weapon() : Item(IT_Weapon)
{
	switch (m_rarity)
	{
	case IR_Normal:
		m_damage = 1 + rand() % 5;
		break;
	case IR_Rare:
		m_damage = 10 + rand() % 10;
		break;
	case IR_Unique:
		m_damage = 30 + rand() % 20;
		break;
	default:
		break;
	}
}

Weapon::~Weapon()
{
}
void Weapon::PrintInfo()
{
	cout << "======================" << endl;
	cout << "[아이템 타입] : 무기" << endl;
	cout << "[공격력] : " << m_damage << endl;
	Item::PrintInfo();
	cout << "======================" << endl;
}
// ============================
//			 Armor
// ============================
Armor::Armor() : Item(IT_Armor)
{
	switch (m_rarity)
	{
	case IR_Normal:
		m_defence = 1 + rand() % 5;
		break;
	case IR_Rare:
		m_defence = 10 + rand() % 10;
		break;
	case IR_Unique:
		m_defence = 30 + rand() % 20;
		break;
	default:
		break;
	}
}

Armor::~Armor()
{
}

void Armor::PrintInfo()
{ 
	cout << "======================" << endl;
	cout << "[아이템 타입] : 방어구" << endl;
	cout << "[방어력] : " << m_defence << endl;
	Item::PrintInfo();
	cout << "======================" << endl;
}