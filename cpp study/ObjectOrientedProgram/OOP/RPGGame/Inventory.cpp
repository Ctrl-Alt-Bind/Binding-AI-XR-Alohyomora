#include "Inventory.h"
#include "Item.h"

// 우린 s_instance 사용할 것이다.
Inventory* Inventory::s_instance = nullptr;

Inventory::Inventory()
{
	for (int i = 0; i < MAX_SLOT; i++)
	{
		m_items[i] = nullptr;
	}
}

Inventory::~Inventory()
{
}

bool Inventory::AddItem(Item* item)
{
	if (item == nullptr) return false;

	int emptySlot = FindEmptySlot();
	if (emptySlot < 0) return false;

	m_items[emptySlot] = item;
	m_itemCount++;
	return true;
}

bool Inventory::RemoveItem(Item* item)
{
	if (item == nullptr) return false;

	int slot = FindItemSlot(item);
	if (slot < 0) return false;

	m_items[slot] = nullptr;
	m_itemCount--;

	return true;
}

int Inventory::FindEmptySlot()
{
	for (int i = 0; i < MAX_SLOT; i++)
	{
		if (m_items[i] == nullptr) return i;
	}

	return -1;
}

int Inventory::FindItemSlot(Item* item)
{
	for (int i = 0; i < MAX_SLOT; i++)
	{
		if (m_items[i] == item) return i;
	}
	return -1;
}

Item* Inventory::GetItemAtSlot(int slot)
{
	if (slot < 0 || slot >= MAX_SLOT) return nullptr;

	return m_items[slot];
}

void Inventory::Clear()
{
	for (int i = 0; i < MAX_SLOT; i++)
	{
		if (m_items[i] != nullptr) {
			delete m_items[i];
			m_items[i] = nullptr;
		}
	}	
}

Inventory* Inventory::GetInstance()
{	
	if (s_instance == nullptr)
		s_instance = new Inventory();

	return s_instance;
}
