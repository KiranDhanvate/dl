#include <iostream>
#include <stack>
#include <omp.h>
using namespace std;

struct Node {
    int data;
    Node *left, *right;

    Node(int v) {
        data = v;
        left = right = NULL;
    }
};

void parallelDFS(Node* root) {

    stack<Node*> s;
    s.push(root);

    while (!s.empty()) {

        int size = s.size();

        #pragma omp parallel for
        for (int i = 0; i < size; i++) {

            Node* curr;

            #pragma omp critical
            {
                curr = s.top();
                s.pop();

                cout << curr->data << " ";

                if (curr->right)
                    s.push(curr->right);

                if (curr->left)
                    s.push(curr->left);
            }
        }
    }
}

int main() {

    // Sample Tree
    Node* root = new Node(1);

    root->left = new Node(2);
    root->right = new Node(3);

    root->left->left = new Node(4);
    root->left->right = new Node(5);

    root->right->left = new Node(6);
    root->right->right = new Node(7);

    cout << "Parallel DFS: ";

    parallelDFS(root);

    return 0;
}
