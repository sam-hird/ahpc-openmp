/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"



float aw1, aw2; /*  accelerate weighting factors */


const float c_sq = 1.f / 3.f; /* square of speed of sound */
const float c_sq_inv    = 3.f;
const float c_sq_sq_inv = 4.5f;
const float w0 = 4.f / 9.f;  /* weighting factor */
const float w1 = 1.f / 9.f;  /* weighting factor */
const float w2 = 1.f / 36.f; /* weighting factor */

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float* speeds[NSPEEDS];
} t_speed;









/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);










/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */

  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */

  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* compute accelerate weighting factors */
  aw1 = params.density * params.accel / 9.f;
  aw2 = params.density * params.accel / 36.f;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    av_vels[tt] = timestep(params, cells, tmp_cells, obstacles);

    //-------------pointer swap-----------
    t_speed* tempPointer = tmp_cells;
    tmp_cells = cells;
    cells = tempPointer;

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

float timestep(const t_param params,  t_speed* restrict cells,  t_speed* restrict tmp_cells, int* restrict obstacles)
{
  


  __assume((params.nx)%2==0);
  __assume((params.ny)%2==0);
  //--------accelerate flow----------

  float tot_cells = 0.f;    /* no. of cells used in calculation */
  float tot_u = 0.f;        /* accumulated magnitudes of velocity for each cell */

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;
  #pragma ivdep
  #pragma vector aligned
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    __assume_aligned(cells->speeds[3],64);
    __assume_aligned(cells->speeds[6],64);
    __assume_aligned(cells->speeds[7],64);
    if (!obstacles[ii + jj * params.nx]
        && (cells->speeds[3][ii + jj * params.nx] - aw1) > 0.f
        && (cells->speeds[6][ii + jj * params.nx] - aw2) > 0.f
        && (cells->speeds[7][ii + jj * params.nx] - aw2) > 0.f)
    {
      /* increase 'east-side' densities */
      /* decrease 'west-side' densities */
      cells->speeds[1][ii + jj * params.nx] +=  aw1;
      cells->speeds[3][ii + jj * params.nx] -=  aw1;
      cells->speeds[5][ii + jj * params.nx] +=  aw2;
      cells->speeds[6][ii + jj * params.nx] -=  aw2;
      cells->speeds[7][ii + jj * params.nx] -=  aw2;
      cells->speeds[8][ii + jj * params.nx] +=  aw2;
    }
  }

  //outer loop is parallel, each thread gets an equal number of rows 
  #pragma omp parallel for schedule(static)
  for (int jj = 0; jj < params.ny; ++jj)
  {
    //each thread has a local version of these sum variables that will be added at the end
    float tot_u1 = 0.f;
    float tot_cells1 = 0.f;

    //inner loop is vectorised using aligned data
    #pragma vector aligned
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ++ii)
    {          
      const int currentIndex = ii + jj * params.nx;
      /* determine neighbours with wrap around */
      const int y_n = (jj + 1) % params.ny;
      const int x_e = (ii + 1) % params.nx;
      const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      /* if the cell contains an obstacle */
      if (obstacles[currentIndex])
      {
        //-----------propagate + rebound---------------
        /* propagate densities from neighbouring cells, following
        ** mirrored directions of travel and writing into
        ** scratch space grid */
        tmp_cells->speeds[0][currentIndex] = cells->speeds[0][ii  + jj *params.nx]; /* central cell, no movement */
        tmp_cells->speeds[1][currentIndex] = cells->speeds[3][x_e + jj *params.nx]; /* west */
        tmp_cells->speeds[2][currentIndex] = cells->speeds[4][ii  + y_n*params.nx]; /* south */
        tmp_cells->speeds[3][currentIndex] = cells->speeds[1][x_w + jj *params.nx]; /* east */
        tmp_cells->speeds[4][currentIndex] = cells->speeds[2][ii  + y_s*params.nx]; /* north */
        tmp_cells->speeds[5][currentIndex] = cells->speeds[7][x_e + y_n*params.nx]; /* south-west */
        tmp_cells->speeds[6][currentIndex] = cells->speeds[8][x_w + y_n*params.nx]; /* south-east */
        tmp_cells->speeds[7][currentIndex] = cells->speeds[5][x_w + y_s*params.nx]; /* north-east */
        tmp_cells->speeds[8][currentIndex] = cells->speeds[6][x_e + y_s*params.nx]; /* north-west */
      } else {
        //-------propagate + collision + av_vels-------

        //avoid bad access pattern by copying all the relevant speeds into an array
        const float currentSpeed0 = cells->speeds[0][ii  +  jj*params.nx];
        const float currentSpeed1 = cells->speeds[1][x_w +  jj*params.nx];
        const float currentSpeed2 = cells->speeds[2][ii  + y_s*params.nx];
        const float currentSpeed3 = cells->speeds[3][x_e +  jj*params.nx];
        const float currentSpeed4 = cells->speeds[4][ii  + y_n*params.nx];
        const float currentSpeed5 = cells->speeds[5][x_w + y_s*params.nx];
        const float currentSpeed6 = cells->speeds[6][x_e + y_s*params.nx];
        const float currentSpeed7 = cells->speeds[7][x_e + y_n*params.nx];
        const float currentSpeed8 = cells->speeds[8][x_w + y_n*params.nx];

        const float local_density = currentSpeed0 + currentSpeed1 + currentSpeed2 
                                  + currentSpeed3 + currentSpeed4 + currentSpeed5 
                                  + currentSpeed6 + currentSpeed7 + currentSpeed8;
        /* compute x velocity component */
        const float u_x = ( currentSpeed1
                    - currentSpeed3
                    + currentSpeed5
                    - currentSpeed6
                    - currentSpeed7
                    + currentSpeed8)
                     / local_density;
        /* compute y velocity component */
        const float u_y = ( currentSpeed2
                    - currentSpeed4
                    + currentSpeed5
                    + currentSpeed6
                    - currentSpeed7
                    - currentSpeed8)
                     / local_density;

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;

        /*pre-compute some parts to avoid many divisions */
        const float u_over_c_sq = 0.5f * u_sq * c_sq_inv;
        /* equilibrium densities */

        
        /* zero velocity density: weight w0 */
        const float d_equ0 = w0 * local_density * (1.f - u_over_c_sq);

        /* axis speeds: weight w1 */
        const float d_equ1 = w1 * local_density * (1.f +   u_x  * c_sq_inv + (  u_x  *   u_x ) * c_sq_sq_inv - u_over_c_sq);
        const float d_equ2 = w1 * local_density * (1.f +   u_y  * c_sq_inv + (  u_y  *   u_y ) * c_sq_sq_inv - u_over_c_sq);
        const float d_equ3 = w1 * local_density * (1.f + (-u_x) * c_sq_inv + ((-u_x) * (-u_x)) * c_sq_sq_inv - u_over_c_sq);
        const float d_equ4 = w1 * local_density * (1.f + (-u_y) * c_sq_inv + ((-u_y) * (-u_y)) * c_sq_sq_inv - u_over_c_sq);
        /* diagonal speeds: weight w2 */
        const float d_equ5 = w2 * local_density * (1.f + ( u_x + u_y) * c_sq_inv + (( u_x + u_y) * ( u_x + u_y)) * c_sq_sq_inv - u_over_c_sq);
        const float d_equ6 = w2 * local_density * (1.f + (-u_x + u_y) * c_sq_inv + ((-u_x + u_y) * (-u_x + u_y)) * c_sq_sq_inv - u_over_c_sq);
        const float d_equ7 = w2 * local_density * (1.f + (-u_x - u_y) * c_sq_inv + ((-u_x - u_y) * (-u_x - u_y)) * c_sq_sq_inv - u_over_c_sq);
        const float d_equ8 = w2 * local_density * (1.f + ( u_x - u_y) * c_sq_inv + (( u_x - u_y) * ( u_x - u_y)) * c_sq_sq_inv - u_over_c_sq);

        /* local density total */
        float av_density = 0.f;
        
        tmp_cells->speeds[0][currentIndex] = currentSpeed0 + params.omega * (d_equ0 - currentSpeed0);
        av_density += tmp_cells->speeds[0][currentIndex];
        tmp_cells->speeds[1][currentIndex] = currentSpeed1 + params.omega * (d_equ1 - currentSpeed1);
        av_density += tmp_cells->speeds[1][currentIndex];
        tmp_cells->speeds[2][currentIndex] = currentSpeed2 + params.omega * (d_equ2 - currentSpeed2);
        av_density += tmp_cells->speeds[2][currentIndex];
        tmp_cells->speeds[3][currentIndex] = currentSpeed3 + params.omega * (d_equ3 - currentSpeed3);
        av_density += tmp_cells->speeds[3][currentIndex];
        tmp_cells->speeds[4][currentIndex] = currentSpeed4 + params.omega * (d_equ4 - currentSpeed4);
        av_density += tmp_cells->speeds[4][currentIndex];
        tmp_cells->speeds[5][currentIndex] = currentSpeed5 + params.omega * (d_equ5 - currentSpeed5);
        av_density += tmp_cells->speeds[5][currentIndex];
        tmp_cells->speeds[6][currentIndex] = currentSpeed6 + params.omega * (d_equ6 - currentSpeed6);
        av_density += tmp_cells->speeds[6][currentIndex];
        tmp_cells->speeds[7][currentIndex] = currentSpeed7 + params.omega * (d_equ7 - currentSpeed7);
        av_density += tmp_cells->speeds[7][currentIndex];
        tmp_cells->speeds[8][currentIndex] = currentSpeed8 + params.omega * (d_equ8 - currentSpeed8);
        av_density += tmp_cells->speeds[8][currentIndex];
        

        /* x-component of velocity */
        const float u_x2 = ( tmp_cells->speeds[1][currentIndex]
              - tmp_cells->speeds[3][currentIndex]
              - tmp_cells->speeds[6][currentIndex]
              - tmp_cells->speeds[7][currentIndex]
              + tmp_cells->speeds[5][currentIndex]
              + tmp_cells->speeds[8][currentIndex])
               / av_density;
        /* compute y velocity component */
        const float u_y2 = ( tmp_cells->speeds[2][currentIndex]
              - tmp_cells->speeds[4][currentIndex]
              + tmp_cells->speeds[5][currentIndex]
              + tmp_cells->speeds[6][currentIndex]
              - tmp_cells->speeds[7][currentIndex]
              - tmp_cells->speeds[8][currentIndex])
               / av_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u1 += sqrtf((u_x2 * u_x2) + (u_y2 * u_y2));
        /* increase counter of inspected cells */
        tot_cells1 += 1.f;
      }
    }
    //threadsafe increments
    #pragma omp atomic
    tot_u += tot_u1;
    #pragma omp atomic
    tot_cells += tot_cells1;
  }  
  return tot_u / tot_cells;
}


int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));
  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->ny));
  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->maxIters));
  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->density));
  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->accel));
  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->omega));
  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);
  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed));
  (*cells_ptr)->speeds[0] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*cells_ptr)->speeds[1] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*cells_ptr)->speeds[2] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*cells_ptr)->speeds[3] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*cells_ptr)->speeds[4] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*cells_ptr)->speeds[5] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*cells_ptr)->speeds[6] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*cells_ptr)->speeds[7] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*cells_ptr)->speeds[8] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed));
  (*tmp_cells_ptr)->speeds[0] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*tmp_cells_ptr)->speeds[1] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*tmp_cells_ptr)->speeds[2] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*tmp_cells_ptr)->speeds[3] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*tmp_cells_ptr)->speeds[4] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*tmp_cells_ptr)->speeds[5] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*tmp_cells_ptr)->speeds[6] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*tmp_cells_ptr)->speeds[7] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);
  (*tmp_cells_ptr)->speeds[8] = (float*)_mm_malloc(sizeof(float) * params->nx * params->ny,64);

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)->speeds[0][ii  + jj*params->nx] = w0;
      /* axis directions */
      (*cells_ptr)->speeds[1][ii  + jj*params->nx] = w1;
      (*cells_ptr)->speeds[2][ii  + jj*params->nx] = w1;
      (*cells_ptr)->speeds[3][ii  + jj*params->nx] = w1;
      (*cells_ptr)->speeds[4][ii  + jj*params->nx] = w1;
      /* diagonals */
      (*cells_ptr)->speeds[5][ii  + jj*params->nx] = w2;
      (*cells_ptr)->speeds[6][ii  + jj*params->nx] = w2;
      (*cells_ptr)->speeds[7][ii  + jj*params->nx] = w2;
      (*cells_ptr)->speeds[8][ii  + jj*params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii  + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {

      total += cells->speeds[0][ii + jj * params.nx];
      total += cells->speeds[1][ii + jj * params.nx];
      total += cells->speeds[2][ii + jj * params.nx];
      total += cells->speeds[3][ii + jj * params.nx];
      total += cells->speeds[4][ii + jj * params.nx];
      total += cells->speeds[5][ii + jj * params.nx];
      total += cells->speeds[6][ii + jj * params.nx];
      total += cells->speeds[7][ii + jj * params.nx];
      total += cells->speeds[8][ii + jj * params.nx];
      
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj * params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;
        local_density += cells->speeds[0][ii + jj * params.nx];
        local_density += cells->speeds[1][ii + jj * params.nx];
        local_density += cells->speeds[2][ii + jj * params.nx];
        local_density += cells->speeds[3][ii + jj * params.nx];
        local_density += cells->speeds[4][ii + jj * params.nx];
        local_density += cells->speeds[5][ii + jj * params.nx];
        local_density += cells->speeds[6][ii + jj * params.nx];
        local_density += cells->speeds[7][ii + jj * params.nx];
        local_density += cells->speeds[8][ii + jj * params.nx];

        /* compute x velocity component */
        u_x = (cells->speeds[1][ii + jj * params.nx]
               + cells->speeds[5][ii + jj * params.nx]
               + cells->speeds[8][ii + jj * params.nx]
               - (cells->speeds[3][ii + jj * params.nx]
                  + cells->speeds[6][ii + jj * params.nx]
                  + cells->speeds[7][ii + jj * params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells->speeds[2][ii + jj * params.nx]
               + cells->speeds[5][ii + jj * params.nx]
               + cells->speeds[6][ii + jj * params.nx]
               - (cells->speeds[4][ii + jj * params.nx]
                  + cells->speeds[7][ii + jj * params.nx]
                  + cells->speeds[8][ii + jj * params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        
        local_density += cells->speeds[0][ii + jj * params.nx];
        local_density += cells->speeds[1][ii + jj * params.nx];
        local_density += cells->speeds[2][ii + jj * params.nx];
        local_density += cells->speeds[3][ii + jj * params.nx];
        local_density += cells->speeds[4][ii + jj * params.nx];
        local_density += cells->speeds[5][ii + jj * params.nx];
        local_density += cells->speeds[6][ii + jj * params.nx];
        local_density += cells->speeds[7][ii + jj * params.nx];
        local_density += cells->speeds[8][ii + jj * params.nx];

        /* x-component of velocity */
        float u_x = (cells->speeds[1][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj * params.nx]
                      + cells->speeds[8][ii + jj * params.nx]
                      - (cells->speeds[3][ii + jj * params.nx]
                         + cells->speeds[6][ii + jj * params.nx]
                         + cells->speeds[7][ii + jj * params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speeds[2][ii + jj * params.nx]
                      + cells->speeds[5][ii + jj * params.nx]
                      + cells->speeds[6][ii + jj * params.nx]
                      - (cells->speeds[4][ii + jj * params.nx]
                         + cells->speeds[7][ii + jj * params.nx]
                         + cells->speeds[8][ii + jj * params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}