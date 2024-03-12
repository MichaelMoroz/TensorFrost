#pragma once
#include <regex>
#include <sstream>
#include <string>
#include <format>
#include "IR/KernelGen.h"
#include "Tensor/Tensor.h"

namespace TensorFrost {

string GetNodeName(const Node* node,  bool compact = false);
void GenerateNodeNames(const IR& ir);

string GenerateC(Program* program);

using ArgumentNames = map<ArgID, string>;

class Line {
 public:
	string left;
	string expression;
	string right;
	string name;
	int indent;
	bool needs_parenthesis = false;
	float cost = 0;

	Line(string left, string expression, string right, string name, bool needs_parenthesis = false, float cost = 0, int indent = 0)
	    : left(left), right(right), name(name), indent(indent), expression(expression), needs_parenthesis(needs_parenthesis), cost(cost) {}

	Line(int indent, string expression)
	    : indent(indent), expression(expression), needs_parenthesis(false), cost(0), left(""), right(""), name("") {}
};

class CodeGenerator {
 public:
	list<Line*> lines;
	map<Node*, string> custom_generated_code_;

	CodeGenerator() = default;

	virtual Line* GenerateLine(Node* node, map<Node*, int> offsets, map<Node*, int> variables) = 0;
	virtual void GenerateArgumentNames(ArgumentManager& args, map<Node*, int> variables) = 0;

	void GenerateKernelLines(const IR* ir, const Node* cluster,
	                         const Kernel* kernel);
	void Compactify();
	string GetFinalCode();
};

string AddIndent(const string& input, const string& indent);


}  // namespace TensorFrost