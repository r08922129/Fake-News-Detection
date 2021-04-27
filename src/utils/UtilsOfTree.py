from collections import deque
import re

def core_nlp_tree_to_json(tree):

    symbols_counts = {}
    out = {}

    queue = deque()
    queue.append(tree)
    
    while queue:
        tree = queue.popleft()
        node = tree.value
        if node != "ROOT":
            node = "{}-{}".format(node, symbols_counts[node])
        
        if len(tree.child):
            out[node] = []
            for child in tree.child:
                
                if child.value not in symbols_counts:
                    symbols_counts[child.value] = 1
                else:
                    symbols_counts[child.value] += 1
                    
                if len(child.child):
                    # child is not a terminal word
                    out[node].append("{}-{}".format(child.value, symbols_counts[child.value]))
                else:
                    out[node].append(child.value)
                queue.append(child)
    return out

def collect_productions_from_json_tree(tree):

    def _getProduction(self, lhs, rhs):
        return f"{lhs} -> {rhs)}"

    def dfs(root, tree, productions):

        if root in tree:
            if tree[root][0] not in tree:
                rhs = "[END]"
            else:
                rhs = " ".join([re.sub(r"-/d+", '', child) for child in tree[root]])

            production = _getProduction(lhs, rhs)
            if production not in productions:
                productions.add(production)

            for child in tree[root]:
                dfs(child, tree, productions)
    productions = set()
    dfs("ROOT", tree, productions)
    return productions
