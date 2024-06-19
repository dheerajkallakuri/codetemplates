class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def construct_bst(preorder):
    if not preorder:
        return None
    
    root = TreeNode(preorder[0])
    stack = [root]
    
    for value in preorder[1:]:
        node = TreeNode(value)
        if value < stack[-1].value:
            stack[-1].left = node
        else:
            while stack and stack[-1].value < value:
                last = stack.pop()
            last.right = node
        stack.append(node)
    
    return root

def find_lca(root, n1, n2):
    while root:
        if root.value > n1 and root.value > n2:
            root = root.left
        elif root.value < n1 and root.value < n2:
            root = root.right
        else:
            return root.value
    return None

def BinaryChallenge(strArr):
    preorder = list(map(int, strArr[0][1:-1].split(', ')))
    n1 = int(strArr[1])
    n2 = int(strArr[2])
    
    root = construct_bst(preorder)
    return find_lca(root, n1, n2)

# Example usage:
strArr = ["[10, 5, 1, 7, 40, 50]", "1", "7"]
print(BinaryChallenge(strArr))  # Output should be 5
