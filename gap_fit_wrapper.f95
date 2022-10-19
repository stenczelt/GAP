! HND XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
! HND X
! HND X   GAP (Gaussian Approximation Potental)
! HND X
! HND X
! HND X   Portions of GAP were written by Albert Bartok-Partay, Gabor Csanyi,
! HND X   Sascha Klawohn, and Tamas K Stenczel. Copyright 2006-2022.
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

#include "error.inc"

! --------------------------------------------------------------------------------
! GAP FIT Wrapper module
! --------------------------------------------------------------------------------
! Allows the use of GAP-FIT as a ligrary for external tools. Exposes the fitting
! procedure through a subroutine, allowing for OMP & MPI parallel usage as well.
!
!
module gap_fit_wrapper_module

  implicit none
  private

  !----------------------------------------------------------------------------
  ! Public Routines
  public :: gap_fit_wrapper

  !----------------------------------------------------------------------------
  ! Everything else private

contains
  ! PUBLIC ROUTINES
  subroutine gap_fit_wrapper(command_line, mpi_communicator, output_unit)
    !% Wrapper for GAP-FIT
    !%
    !% Assumes that another program is in charge, which does other work
    !% as well, and that calls this multiple times during it's execution.
    !%
    !% Written by Tamas K. Stenczel, 20/09/2022

    use gap_fit_module, only : CMD_STR_LENGTH, gap_fit_main_logic, gap_fit
    use system_module, only : PRINT_SILENT, PRINT_NORMAL, system_initialise
    use MPI_context_module, only : MPI_context, Initialise

    implicit none

    ! call params
    character(len=CMD_STR_LENGTH), intent(in) :: command_line
!    character(len=CMD_STR_LENGTH), intent(in) :: param_filename
    integer, intent(in), optional :: mpi_communicator
    integer, intent(in), optional :: output_unit

    ! temporary internals
    type(gap_fit) :: main_gap_fit

    ! saved internals
    type(MPI_context), save :: mpi_glob
    logical, save :: first_run=.true.

    ! Initialisation of system & MPI - can be done only once
    if (first_run) then
      ! system & IO
      call system_initialise(verbosity=PRINT_SILENT, mainlog_unit=output_unit)
      ! initialise MPI with the communicator given
      call Initialise(mpi_glob, communicator=mpi_communicator)
    else
      ! todo: include checks for MPI comm & output unit being unchanged
    end if

    ! pass command line
    main_gap_fit%command_line = command_line

    ! do the actual fitting
    call gap_fit_main_logic(main_gap_fit)

  end subroutine gap_fit_wrapper

end module gap_fit_wrapper_module


