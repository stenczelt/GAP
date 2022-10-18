! HND XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
! HND X
! HND X   GAP (Gaussian Approximation Potental)
! HND X
! HND X
! HND X   Portions of GAP were written by Albert Bartok-Partay, Gabor Csanyi,
! HND X   and Sascha Klawohn. Copyright 2006-2021.
! HND X
! HND X   Portions of GAP were written by Noam Bernstein as part of
! HND X   his employment for the U.S. Government, and are not subject
! HND X   to copyright in the USA.
! HND X
! HND X   GAP is published and distributed under the
! HND X      Academic Software License v1.0 (ASL)
! HND X
! HND X   GAP is distributed in the hope that it will be useful for non-commercial
! HND X   academic research, but WITHOUT ANY WARRANTY; without even the implied
! HND X   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! HND X   ASL for more details.
! HND X
! HND X   You should have received a copy of the ASL along with this program
! HND X   (e.g. in a LICENSE.md file); if not, you can write to the original licensors,
! HND X   Gabor Csanyi or Albert Bartok-Partay. The ASL is also published at
! HND X   http://github.com/gabor1/ASL
! HND X
! HND X   When using this software, please cite the following reference:
! HND X
! HND X   A. P. Bartok et al Physical Review Letters vol 104 p136403 (2010)
! HND X
! HND X   When using the SOAP kernel or its variants, please additionally cite:
! HND X
! HND X   A. P. Bartok et al Physical Review B vol 87 p184115 (2013)
! HND X
! HND XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

program gap_fit_program

  use system_module, only : system_initialise, PRINT_NORMAL, system_finalise
  use gap_fit_module, only : gap_fit, gap_fit_main_program, gap_fit_read_command_line

  implicit none

  ! internals
  type(gap_fit) :: main_gap_fit

  ! program content


  ! initialise the system & MPI
  call system_initialise(verbosity=PRINT_NORMAL, enable_timing=.false.)

  ! read the command line parameters
  call gap_fit_read_command_line(main_gap_fit)

  ! call the library routine to do the actual work
  call gap_fit_main_program(main_gap_fit)

  ! finalise the program
  call system_finalise()

end program gap_fit_program
