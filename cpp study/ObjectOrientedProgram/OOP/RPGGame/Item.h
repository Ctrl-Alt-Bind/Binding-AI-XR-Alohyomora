#pragma once

// Item
// - Weapon
// - Armor
// - Consumable

#include "enums.h"

class Item
{
protected:
	// Item(); // 기본 생성자
	Item(ItemType itemType);
public:
	virtual ~Item();

public:
	// void PrintInfo();
	virtual void PrintInfo(); // 가상 함수, 상속한 아이의 태생적인 정보에 따라 출력
	ItemType GetItemType() { return m_type; }

protected:
	int m_itenId = 0;
	int m_itemCount = 0; // 물약, 버프 100개 이상
	ItemRarity m_rarity = IR_None;
	ItemType m_type = IT_None;
};

class Weapon : public Item
{
public:
	Weapon();
	virtual ~Weapon();

	virtual void PrintInfo() override;

	void SetDamage(int damage) { m_damage = damage; }
	int GetDamage() { return m_damage; }
private:
	int m_damage = 0;
};

class Armor : public Item
{
public:
	Armor();
	virtual ~Armor();
	
	virtual void PrintInfo() override;

	void SetDefence(int defence) { m_defence = defence; }
	int GetDefence() { return m_defence; }
private:
	int m_defence = 0;
};