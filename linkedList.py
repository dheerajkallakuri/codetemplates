class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def prepend(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def delete_value(self, value):
        current = self.head
        if current and current.data == value:
            self.head = current.next
            current = None
            return
        prev = None
        while current and current.data != value:
            prev = current
            current = current.next
        if current is None:
            return
        prev.next = current.next
        current = None

    def display(self):
        current = self.head
        while current:
            print(current.data, end=' -> ')
            current = current.next
        print('None')
    
    #O(n log n)
    def merge_sort(self, head):
        if head is None or head.next is None:
            return head
        
        middle = self.get_middle(head)
        next_to_middle = middle.next
        middle.next = None

        left = self.merge_sort(head)
        right = self.merge_sort(next_to_middle)

        sorted_list = self.sorted_merge(left, right)
        return sorted_list

    def get_middle(self, head):
        if head is None:
            return head

        slow = head
        fast = head

        while fast.next is not None and fast.next.next is not None:
            slow = slow.next
            fast = fast.next.next

        return slow

    def sorted_merge(self, left, right):
        if left is None:
            return right
        if right is None:
            return left

        if left.data <= right.data:
            result = left
            result.next = self.sorted_merge(left.next, right)
        else:
            result = right
            result.next = self.sorted_merge(left, right.next)
        return result

    def sort(self):
        self.head = self.merge_sort(self.head)

# Example usage:
if __name__ == '__main__':
    ll = LinkedList()
    ll.append(3)
    ll.append(2)
    ll.append(1)
    ll.prepend(0)
    ll.delete_value(2)
    ll.display()
    ll.sort()
    ll.display()