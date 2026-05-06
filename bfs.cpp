#include <iostream>
#include <queue>
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

void parallelBFS(Node* root) {
    queue<Node*> q;
    q.push(root);  

    while (!q.empty()) {
        int size = q.size();

        #pragma omp parallel for
        for (int i = 0; i < size; i++) {

            Node* curr;

            #pragma omp critical
            {
                curr = q.front();
                q.pop();

                cout << curr->data << " ";

                if (curr->left)
                    q.push(curr->left);

                if (curr->right)
                    q.push(curr->right);
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

    cout << "Parallel BFS: ";

    parallelBFS(root);

    return 0;
}
