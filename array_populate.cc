#include <cstdio>
#include <cstdlib>

#include "legion.h"

#define ROW 5
#define COL 5

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

enum TASK_ID  {
  TOP_LEVEL_TASK_ID,
  PRINT_LR_TASK_ID,
  GENERATE_RHS_TASK_ID,
  GENERATE_X0_TASK_ID,
  TRIM_ROW_TASK_ID,
  TRIM_FIELD_TASK_ID
};

enum FieldIDs {
  FID_RHS,
  FID_TRIMMED_COL
};

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime)
{
  int field_id[COL];

  Rect<1> elem_rect(Point<1>(0), Point<1>(ROW - 1));
  IndexSpace is = runtime->create_index_space(ctx, Domain::from_rect<1>(elem_rect));
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);

    for(int i = 0; i < COL; i++)  {
      field_id[i] = allocator.allocate_field(sizeof(double));
    }
  }

  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, fs);

  RegionRequirement req(input_lr, READ_WRITE, EXCLUSIVE, input_lr);

  /* specify which fields on logical regions to request */
  for(int i = 0; i < COL; i++)  {
    req.add_field(field_id[i]);
  }

  // req.add_field(field_id[0]);
  // req.add_field(field_id[1]);
  // req.add_field(field_id[2]);
  // req.add_field(field_id[3]);
  // req.add_field(field_id[4]);

  InlineLauncher input_launcher(req);
  PhysicalRegion input_region = runtime->map_region(ctx, input_launcher);
  input_region.wait_until_valid();

  RegionAccessor<AccessorType::Generic, double> region_accessor[COL];

  for(int i = 0; i < COL; i++)  {
    region_accessor[i] = input_region.get_field_accessor(field_id[i]).typeify<double>();
  }

  for(GenericPointInRectIterator<1> pir(elem_rect); pir; pir++) {
    for(int i = 0; i < COL; i++)  {
       region_accessor[i].write(DomainPoint::from_point<1>(pir.p), (rand() % 1000));
    }
  }

  TaskLauncher print_lr_launcher(PRINT_LR_TASK_ID, TaskArgument(field_id, sizeof(field_id)));
  print_lr_launcher.add_region_requirement(RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  for(int i = 0; i < COL; i++)  {
    print_lr_launcher.add_field(0, field_id[i]);
  }
  runtime->execute_task(ctx, print_lr_launcher);


  Rect<1> elem_rect2(Point<1>(0), Point<1>(ROW - 1));
  IndexSpace rhs_is = runtime->create_index_space(ctx, Domain::from_rect<1>(elem_rect2));
  FieldSpace rhd_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, rhd_fs);
    allocator.allocate_field(sizeof(double), FID_RHS);
  }

  LogicalRegion rhs_lr = runtime->create_logical_region(ctx, rhs_is, rhd_fs);

  TaskLauncher generate_rhs_launcher(GENERATE_RHS_TASK_ID, TaskArgument(NULL, 0));
  generate_rhs_launcher.add_region_requirement(
        RegionRequirement(rhs_lr, WRITE_DISCARD, EXCLUSIVE, rhs_lr));
  generate_rhs_launcher.add_field(0, FID_RHS);
  runtime->execute_task(ctx, generate_rhs_launcher);

  /* GENERATE_X0_TASK */
  TaskLauncher generate_x0_task_launcher;
  generate_x0_task_launcher.task_id = GENERATE_X0_TASK_ID;
  generate_x0_task_launcher.add_region_requirement(
    RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  generate_x0_task_launcher.add_field(0, field_id[0]);


  /* TRIM_ROW_TASK */
  TaskLauncher trim_row_task_launcher; //(TRIM_ROW_TASK_ID); //, TaskArgument(NULL, 0));
  trim_row_task_launcher.task_id = TRIM_ROW_TASK_ID;
  trim_row_task_launcher.argument = TaskArgument(NULL, 0);
  // TaskLauncher trim_row_task_launcher(TRIM_ROW_TASK_ID, TaskArgument(field_id, sizeof(field_id)));
  // trim_row_task_launcher.add_future(f_x0);
  trim_row_task_launcher.add_region_requirement(
  RegionRequirement(input_lr, READ_WRITE, EXCLUSIVE, input_lr));
  for(int i = 0; i < COL; i++)  {
    trim_row_task_launcher.add_field(0, field_id[i]);
  }

  Rect<1> launch_bounds_x0(Point<1>(0), Point<1>(ROW - 2));
  Domain launch_domain_x0 = Domain::from_rect<1>(launch_bounds_x0);
  ArgumentMap arg_map_x0;
  for(int i = 0; i < (ROW - 1); i++)
  {
    int input = i + 1;
    arg_map_x0.set_point(DomainPoint::from_point<1>(Point<1>(i)),
                    TaskArgument(&input, sizeof(input)));
  }

  IndexLauncher index_launcher_x0(GENERATE_X0_TASK_ID,
    launch_domain_x0, TaskArgument(NULL, 0), arg_map_x0);
  index_launcher_x0.add_region_requirement(
    RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  index_launcher_x0.add_field(0, field_id[0]);

  FutureMap fm = runtime->execute_index_space(ctx, index_launcher_x0);
  fm.wait_all_results();


  // Rect<1> elem_rect3(Point<1>(0), Point<1>(ROW - 1));
  // IndexSpace trimmed_row = runtime->create_index_space(ctx, Domain::from_rect<1>(elem_rect3));
  // FieldSpace trimmed_col = runtime->create_field_space(ctx);
  // {
  //   FieldAllocator allocator = runtime->create_field_allocator(ctx, trimmed_col);
  //   allocator.allocate_field(sizeof(double), FID_TRIMMED_COL);
  // }
  //
  // LogicalRegion trim_lr = runtime->create_logical_region(ctx, trimmed_row, trimmed_col);
  //
  // TaskLauncher trim_field_task_launcher(TRIM_FIELD_TASK_ID, TaskArgument(NULL, 0));
  // trim_field_task_launcher.add_region_requirement(
  //   RegionRequirement(trim_lr, WRITE_DISCARD, EXCLUSIVE, trim_lr));
  // trim_field_task_launcher.add_field(0, FID_TRIMMED_COL);
  // trim_field_task_launcher.add_region_requirement(
  //   RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  // trim_field_task_launcher.add_field(1, field_id[0]);
  // runtime->execute_task(ctx, trim_field_task_launcher);

  printf("\n Done!\n");
}

double generate_x0_task(const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx, HighLevelRuntime *runtime) {

  int my_rank = task->index_point.point_data[0];
  int input_row_id = *((const int*) task->local_args);
  printf("\n Inside generate_x0_task() #%d | input = %d", my_rank, input_row_id);

  FieldID fid_orig = *(task->regions[0].privilege_fields.begin());
  //
  // int target_row = *((int *) task->args);
  // printf("\n TaskArgument from gxt: #%d", target_row);
  //
  RegionAccessor<AccessorType::Generic, double> acc_orig =
    regions[0].get_field_accessor(fid_orig).typeify<double>();

  printf("\n Printing out current row: \n");
  for(int i = 0; i < ROW; i++) {
    double x = acc_orig.read(DomainPoint::from_point<1>(i));
    printf("\n -> %lf", x);
  }

  double divident = acc_orig.read(DomainPoint::from_point<1>(input_row_id));
  double divisor = acc_orig.read(DomainPoint::from_point<1>(0));
  // double divisor = acc_orig.read(DomainPoint::from_point<1>(target_row - 1));
  double result = (divident/divisor);

  printf("\n %lf %lf %lf", divident, divisor, result);

  printf("\n XO from function #%d: %lf\n", my_rank, result);

  return result;
  //return 0;
}

void trim_row_task(const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx, HighLevelRuntime *runtime) {
  printf("\n Inside trim_row_task()");

  Future f_x = task->futures[0];
  double x0 = f_x.get_result<double>();
  printf("\n From trim_row_task: %lf\n", x0);

  int target_row = *((int *) task->args);
  printf("\n TaskArgument from trt: #%d", target_row);
  //int *trim_field_id = (int *) task->args;

  // int size = *(task->regions[0].privilege_fields.begin());
  // printf("\n Size: %d\n", size);

  /* Get all the field IDs */
  FieldID trim_field_id[COL];
  int tf = *(task->regions[0].privilege_fields.begin());
  // Figure out the other field IDs...
  for(int i = 0; i < COL; i++)
  {
    trim_field_id[i] = tf++;
    printf("\n TFI: %d", trim_field_id[i]);
  }

  // Accessor for the fields
  RegionAccessor<AccessorType::Generic, double> region_accessor[COL];
  for(int i = 0; i < COL; i++)  {
    region_accessor[i] = regions[0].get_field_accessor(trim_field_id[i]).typeify<double>();
  }

  printf("\n Printing values before reduction: \n");
  for(int i = 0; i < ROW; i++)  {
    for(int j = 0; j < COL; j++)  {
      double x = region_accessor[j].read(DomainPoint::from_point<1>(Point<1>(i)));
      printf(" = %lf", x);
    }
    printf("\n");
  }

  for(int i = 0; i < COL; i++)  {
    /* read the columns of row 0 */
    double x = region_accessor[i].read(DomainPoint::from_point<1>(0));
    x = x * x0;
    double y = region_accessor[i].read(DomainPoint::from_point<1>(target_row));
    region_accessor[i].write(DomainPoint::from_point<1>(target_row), (y - x));
  }

  printf("\n Printing out the reduced values: \n");
  for(int i = 0; i < ROW; i++)  {
    for(int j = 0; j < COL; j++)  {
      double x = region_accessor[j].read(DomainPoint::from_point<1>(i));
      printf(" = %lf", x);
    }
    printf("\n");
  }

}

void trim_field_task(const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx, HighLevelRuntime *runtime) {
  printf("\nInside trim_field_task\n");

  FieldID fid_orig = *(task->regions[1].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, double> acc_trim =
    regions[0].get_field_accessor(FID_TRIMMED_COL).typeify<double>();

  RegionAccessor<AccessorType::Generic, double> acc_orig =
    regions[1].get_field_accessor(fid_orig).typeify<double>();

  double first_element = acc_orig.read(DomainPoint::from_point<1>(0));
  double current_element = 0;
  double replacement_element = 0;
  for(int i = 1; i < ROW; i++) {
    current_element = acc_orig.read(DomainPoint::from_point<1>(i));
    replacement_element = current_element / first_element;
    acc_trim.write(DomainPoint::from_point<1>(i), replacement_element);
  }

  printf("\n Printing out the trimmed row: \n");
  for(int i = 0; i < ROW; i++) {
    double x = acc_trim.read(DomainPoint::from_point<1>(i));
    printf("\n -> %lf", x);
  }
}

void generate_rhs_task(const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx, HighLevelRuntime *runtime) {
  printf("\n Inside generate_rhs_task()");

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());

  RegionAccessor <AccessorType::Generic, double> acc =
    regions[0].get_field_accessor(fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  for(GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    acc.write(DomainPoint::from_point<1>(pir.p), rand() % 10);
  }

  printf("\n Filled in random() values into RHS");
}

void print_lr_task(const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx, HighLevelRuntime *runtime) {

  int *field_id = (int *) task->args;

  RegionAccessor<AccessorType::Generic, double> region_accessor[COL];
  for(int i = 0; i < COL; i++)  {
    region_accessor[i] = regions[0].get_field_accessor(field_id[i]).typeify<double>();
  }

  Domain domain = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  Rect <1> rect = domain.get_rect<1>();

  printf("\n Printing Loaded Values:\n");

  for(GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    for(int i = 0; i < COL; i++)  {
      double x = region_accessor[i].read(DomainPoint::from_point<1>(pir.p));
      printf("  > %lf", x);
    }
    printf("\n");
  }
}

int main(int argc, char **argv) {
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID, Processor::LOC_PROC, true, false);
  HighLevelRuntime::register_legion_task<print_lr_task>(PRINT_LR_TASK_ID, Processor::LOC_PROC, true, false);
  HighLevelRuntime::register_legion_task<generate_rhs_task>(GENERATE_RHS_TASK_ID, Processor::LOC_PROC, true, false);
  HighLevelRuntime::register_legion_task<double, generate_x0_task>(GENERATE_X0_TASK_ID, Processor::LOC_PROC, true, true /* index */);
  HighLevelRuntime::register_legion_task<trim_row_task>(TRIM_ROW_TASK_ID, Processor::LOC_PROC, true, false);
  // HighLevelRuntime::register_legion_task<trim_field_task>(TRIM_FIELD_TASK_ID, Processor::LOC_PROC, true, false);

  return HighLevelRuntime::start(argc, argv);
}
