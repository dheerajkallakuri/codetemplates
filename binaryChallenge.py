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

def construct_bst_from_postorder(postorder):
    if not postorder:
        return None
    
    root = TreeNode(postorder[-1])
    stack = [root]
    
    for value in reversed(postorder[:-1]):
        node = TreeNode(value)
        if value > stack[-1].value:
            stack[-1].right = node
        else:
            while stack and stack[-1].value > value:
                last = stack.pop()
            last.left = node
        stack.append(node)
    
    return root

def build_bst_from_inorder(inorder, preorder):
    if not inorder or not preorder:
        return None
    
    root_value = preorder.pop(0)
    root = TreeNode(root_value)
    inorder_index = inorder.index(root_value)
    
    root.left = build_bst_from_inorder_preorder(inorder[:inorder_index], preorder)
    root.right = build_bst_from_inorder_preorder(inorder[inorder_index+1:], preorder)
    
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

def BinaryChallenge(strArr):
    inorder = list(map(int, strArr[0][1:-1].split(', ')))
    preorder = list(map(int, strArr[1][1:-1].split(', ')))
    n1 = int(strArr[2])
    n2 = int(strArr[3])
    
    root = build_bst_from_inorder(inorder, preorder)
    return find_lca(root, n1, n2)

# Example usage:
strArr = ["[10, 5, 1, 7, 40, 50]", "1", "7"]
print(BinaryChallenge(strArr))  # Output should be 5

strArr1 = ["[1, 5, 7, 10, 40, 50]", "[10, 5, 1, 7, 40, 50]", "1", "7"]
print(BinaryChallenge_inorder(strArr))
