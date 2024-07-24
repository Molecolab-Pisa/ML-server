#include "../include/dprec.fh"
module qm2_extern_socket_module
! ----------------------------------------------------------------
! Interface for socket-based ML and ML/MM MD 
!
! Currently supports:
! pure ML
! ML/MM with cutoff for ML-MM electrostatics 
!
! Authors: Patrizia Mazzeo and Edoardo Cignoni
! Based on qm2_extern_orc_module.f
!
! Date: June 2024
!
! ----------------------------------------------------------------

  implicit none

  private
  public :: get_socket_forces

  character(len=*), parameter :: module_name = "qm2_extern_socket_module"

  type socket_nml_type
     integer           :: port
     integer           :: debug
  end type socket_nml_type

contains

  ! --------------------------------------
  ! Get ML energy and forces from socket
  ! -------------------------------------- 
  subroutine get_socket_forces( do_grad, nstep, ntpr_default, id, &
       nqmatoms, qmcoords, qmtypes, nclatoms, clcoords, escf, dxyzqm, dxyzcl, &
       charge, spinmult )

    use qm2_extern_util_module, only: print_results, check_installation
    use constants,              only: CODATA08_AU_TO_KCAL, CODATA08_A_TO_BOHRS, ZERO

    use f90sockets, only : open_socket, close_socket, writebuffer, readbuffer ! functions to connect, read and write via socket
    use, intrinsic :: iso_c_binding

    logical, intent(in) :: do_grad              ! Return gradient/not
    integer, intent(in) :: nstep                ! MD step number
    integer, intent(in) :: ntpr_default         ! frequency of printing
    character(len=3), intent(in) :: id          ! ID number for PIMD or REMD
    integer, intent(in) :: nqmatoms             ! Number of QM atoms
    _REAL_,  intent(in) :: qmcoords(3,nqmatoms) ! QM coordinates
    integer, intent(in) :: qmtypes(nqmatoms)    ! QM atom types (nuclear charge in au)
    integer, intent(in) :: nclatoms             ! Number of MM atoms
    _REAL_,  intent(in) :: clcoords(4,nclatoms) ! MM coordinates and charges in au
    _REAL_, intent(out) :: escf                 ! SCF energy
    _REAL_, intent(out) :: dxyzqm(3,nqmatoms)   ! SCF QM force
    _REAL_, intent(out) :: dxyzcl(3,nclatoms)   ! SCF MM force
    integer, intent(in) :: charge, spinmult     ! Charge and spin multiplicity

    integer, parameter  :: msglen=12                ! length of messages to communicate
    integer             :: socket, inet, port       ! socket id and address of the server
    integer             :: system_data(2)
    character(len=1024) :: host

    character(len=1024)       :: header             ! It's used to read messages
    real(kind=8), allocatable :: msgbuffer(:)       ! It's used to read data

    type(socket_nml_type), save :: socket_nml
    logical, save :: first_call = .true.
    integer :: i
    integer :: printed =-1 ! Used to tell if we have printed this step yet 
                           ! since the same step may be called multiple times
    character(len=*), parameter  :: server='ml-server'
    character(len=256), save :: server_path

    inet = 1
!    port = 3004
!    host = "127.0.0.1"
    host = "localhost"//achar(0)

    ! Setup on first program call
    if ( first_call ) then
      first_call = .false.
      write (6,'(/,a,/)') '   >>> Running ML prediction via Socket <<<'
      call get_namelist( ntpr_default, socket_nml )
      call check_installation( server, id, .true., socket_nml%debug, path=server_path )
      call print_namelist( socket_nml )

      write (6,'(80a)') ('-', i=1,80)
      write (6,'(a)') '   4.  RESULTS'
      write (6,'(80a)') ('-', i=1,80)

    end if

    system_data(1) = nqmatoms
    system_data(2) = nclatoms

    ! open socket connection
    call open_socket(socket,inet,socket_nml%port,host)
    do while (.true.) ! loops forever (or until the wrapper ends!)

      call writebuffer(socket,"model-run   ",msglen)

      ! Sending QM and MM atoms number
      msgbuffer = reshape(system_data,[size(system_data)])
      call writebuffer(socket,msgbuffer,size(system_data))

      ! Seding QM atoms coordinates
      msgbuffer = reshape(qmcoords,[size(qmcoords)])
      call writebuffer(socket,msgbuffer,size(qmcoords))

      ! Seding MM coordinates and charges
      ! and reading MM gradients
      if ( nclatoms > 0 ) then
        msgbuffer = reshape(clcoords,[size(clcoords)])
        call writebuffer(socket,msgbuffer,size(clcoords))
        call readbuffer(socket,msgbuffer,size(dxyzcl))
        dxyzcl = reshape(msgbuffer,shape(dxyzcl))
      endif 

      ! Reading QM gradients
      call readbuffer(socket,msgbuffer,size(dxyzqm))
      dxyzqm = reshape(msgbuffer,shape(dxyzqm))

      ! Reading SCF energy
      call readbuffer(socket,escf)

      call readbuffer(socket,header,msglen)
      if (trim(header).eq."model-fin") exit

    end do

    if ( do_grad ) then
      ! Convert Hartree/Bohr -> kcal/(mol*A)
      dxyzqm(:,:) = dxyzqm(:,:) * CODATA08_AU_TO_KCAL * CODATA08_A_TO_BOHRS
      if ( nclatoms > 0 ) then
         dxyzcl(:,:) = dxyzcl(:,:) * CODATA08_AU_TO_KCAL * CODATA08_A_TO_BOHRS
      end if
    else
      dxyzqm = ZERO
      if ( nclatoms > 0 ) dxyzcl = ZERO
    end if

    escf = escf * CODATA08_AU_TO_KCAL

    call print_results( 'qm2_extern_socket_module', escf, nqmatoms, dxyzqm,&
      socket_nml%debug, nclatoms, dxyzcl )

    call close_socket(socket)

  end subroutine get_socket_forces

  ! -----------------------------------------------
  ! Read socket namelist values from file mdin,
  ! use default values if none are present.
  ! -----------------------------------------------
  subroutine get_namelist( ntpr_default, socket_nml )

    use UtilitiesModule, only: Upcase
    implicit none

    integer, intent(in) :: ntpr_default
    type(socket_nml_type), intent(out) :: socket_nml
    integer :: port, debug
    namelist /socket/ port, debug 
    integer :: ierr

    ! Default values
    port         = 3004
    debug        = 0

    ! Read namelist
    rewind 5
    read(5,nml=socket,iostat=ierr)

    if ( ierr > 0 ) then
       call sander_bomb('get_namelist (qm2_extern_socket_module)', &
            '&socket namelist read error', &
            'Please check your input.')
    else if ( ierr < 0 ) then
       write(6,'(a,/,a)') '&socket namelist read encountered end of file', &
            'Please check your input if the calculation encounters a problem'
    end if

    ! Assign namelist values to socket_nml data type
    socket_nml%port        = port  
    socket_nml%debug       = debug

  end subroutine get_namelist

  ! --------------------------------
  ! Print Socket namelist settings
  ! --------------------------------
  subroutine print_namelist( socket_nml )

    implicit none
    type(socket_nml_type), intent(in) :: socket_nml

    write(6, '(/,a)')      '| &socket'
    write(6, '(a,i4)')     '|   port         = ', socket_nml%port
    write(6, '(a,i2)')     '|   debug        = ', socket_nml%debug
    write(6,'(a)')         '| /'

  end subroutine print_namelist
end module qm2_extern_socket_module
