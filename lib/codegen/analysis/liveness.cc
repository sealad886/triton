#include <iostream>
#include <climits>
#include <unordered_set>
#include "triton/codegen/instructions.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/transform/cts.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/instructions.h"
#include "triton/ir/value.h"
#include "triton/ir/utils.h"

namespace triton{
namespace codegen{
namespace analysis{

inline bool is_loop_latch(ir::phi_node *phi, ir::instruction *terminator){
  if(phi->get_parent() != terminator->get_parent())
    return false;
  if(auto *br = dynamic_cast<ir::cond_branch_inst*>(terminator))
    return br->get_true_dest() == phi->get_parent()
           || br->get_false_dest() == phi->get_parent();
  else if(dynamic_cast<ir::uncond_branch_inst*>(terminator))
    return false;
  else
    throw std::runtime_error("unreachable");
}

void liveness::extract_double_bufferable(ir::instruction *i) {
  auto* phi = dynamic_cast<ir::phi_node*>(i);
  if(!phi || phi->get_num_incoming() != 2)
    return;
  ir::basic_block *block_0 = phi->get_incoming_block(0);
  ir::basic_block *block_1 = phi->get_incoming_block(1);
  ir::instruction *terminator_0 = block_0->get_inst_list().back();
  ir::instruction *terminator_1 = block_1->get_inst_list().back();
  bool is_latch_0 = is_loop_latch(phi, terminator_0);
  bool is_latch_1 = is_loop_latch(phi, terminator_1);
  ir::value *value_0 = phi->get_incoming_value(0);
  ir::value *value_1 = phi->get_incoming_value(1);
  ir::instruction *i_0 = dynamic_cast<ir::instruction*>(value_0);
  ir::instruction *i_1 = dynamic_cast<ir::instruction*>(value_1);
  if(!i_0 || !i_1 || storage_info.at(i_0->get_id()).first != codegen::SHARED || storage_info.at(i_1->get_id()).first != codegen::SHARED)
    return;
  if(is_latch_1)
    double_[value_0] = double_buffer_info_t{value_1, phi};
  if(is_latch_0)
    double_[value_1] = double_buffer_info_t{value_0, phi};
}


// connected components
bool is_trans(ir::value *v) {
  if(dynamic_cast<ir::trans_inst *>(v)) {
    return true;
  }
  if(auto *phi = dynamic_cast<ir::instruction *>(v)) {
    bool result = true;
    for(ir::value *op: phi->ops())
      result = result && is_trans(op);
    return result;
  }
  return false;
}


bool liveness::do_pad(ir::value *x) {
  // alignment for matrix product
  if(auto* dot = dynamic_cast<ir::dot_inst*>(x)) {
    // a
    ir::value *a = dot->get_operand(0);
    ir::value *b = dot->get_operand(1);
    size_t a_previous = pad_[a];
    size_t b_previous = pad_[b];
    auto a_order = layouts_->get(a)->order;
    auto b_order = layouts_->get(b)->order;
    bool a_row = is_trans(a) ^ (a_order[0] == 1);
    bool b_row = is_trans(b) ^ (b_order[0] == 1);
    auto a_shapes = a->get_type()->get_tile_shapes();
    auto b_shapes = b->get_type()->get_tile_shapes();
    pad_[a] = std::max<int>(pad_[a], (24 - a_shapes[a_row ? 0 : 1]) % 32);
    pad_[b] = std::max<int>(pad_[b], (24 - b_shapes[b_row ? 1 : 0]) % 32);
    return a_previous != pad_[a] || b_previous != pad_[b];
  }
  // padding for trans
  if(auto* trans = dynamic_cast<ir::trans_inst*>(x)) {
    ir::value *op = trans->get_operand(0);
    size_t previous = pad_[op];
    pad_[op] = std::max(pad_[op], pad_[x]);
    return previous != pad_[op];
  }
  // padding for copy to shared
  if(auto* cts = dynamic_cast<ir::copy_to_shared_inst*>(x)) {
    auto cts_order = layouts_->get(cts)->order;
    ir::value *arg = cts->get_operand(0);
    auto arg_order = layouts_->get(arg)->order;
    size_t previous = pad_[cts];
    if(cts_order != arg_order)
      pad_[cts] = std::max<int>(pad_[cts], 4);
    return pad_[cts] != previous;
  }
  // padding for phi-nodes
  if(auto* phi = dynamic_cast<ir::phi_node*>(x)) {
    bool has_changed = false;
    for(unsigned i = 0; i < phi->get_num_incoming(); i++){
      ir::value* op = phi->get_operand(i);
      size_t previous = pad_[op];
      pad_[op] = std::max(pad_[op], pad_[phi]);
      has_changed |= previous != pad_[op];
    }
    return has_changed;
  }
  // default -- no padding
  size_t previous = pad_[x];
  pad_[x] = std::max<int>(previous, 0);
  return pad_[x] != previous;
}

unsigned liveness::num_bytes(ir::value *x) {
  unsigned num_bytes = x->get_type()->get_primitive_size_in_bits() / 8;
  unsigned pad = pad_.at(x);
  if(pad > 0){
    unsigned ld = x->get_type()->get_tile_shapes()[layouts_->get(x)->order[0]];
    num_bytes += pad * num_bytes / ld;
  }
  if(has_double(x))
    num_bytes *= 2;
  return num_bytes;
}

// Entry point
void liveness::run(ir::module &mod) {
  double_.clear();
  indices.clear();
  pad_.clear();
  intervals_.clear();

  // Create set of pair of values that can be double-buffered
  ir::for_each_instruction(mod, [this](ir::instruction* i) {
    this->extract_double_bufferable(i);
  });

  // Padding information
  bool has_changed;
  do{
    has_changed = false;
    ir::for_each_value(mod, [this, &has_changed](ir::value* v){
      has_changed |= this->do_pad(v);
    });
  }while(has_changed);


  // connected components
  for(auto &x: layouts_->get_all()) {
    layout_t*& layout = x.second;
    if(layout->type != SHARED)
      continue;
    for(ir::value *v: layout->values)
      layout->size = std::max<int>(layout->size, num_bytes(v));
  }

  // Assigns index to each instruction
  for(ir::function *fn: mod.get_function_list()){
    slot_index index = 0;
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *instr: block->get_inst_list()){
      index += 1;
      indices.insert({instr, index});
    }
  }

  for(auto &x: layouts_->get_all()) {
    layout_t* layout = x.second;
    if(layout->type != SHARED)
      continue;
    // users
    std::set<ir::value*> users;
    for(ir::value *v: layout->values){
      users.insert(v);
      for(ir::user *u: v->get_users())
        users.insert(u);
    }
    // compute intervals
    unsigned start = INT32_MAX;
    unsigned end = 0;
    for(ir::value *u: users){
      start = std::min(start, indices.at(u));
      end = std::max(end, indices.at(u));
    }
    intervals_[layout] = segment{start, end};
  }



}

}
}
}
