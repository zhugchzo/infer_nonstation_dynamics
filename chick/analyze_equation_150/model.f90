!----------------------------------------------------------------------
!----------------------------------------------------------------------
!   model: Fortran script to specify model for AUTO
!----------------------------------------------------------------------
!----------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      DOUBLE PRECISION x,b,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10
       x=U(1)
       b=PAR(1)
       a1=PAR(2)
       a2=PAR(3)
       a3=PAR(4)
       a4=PAR(5)
       a5=PAR(6)
       a6=PAR(7)
       a7=PAR(8)
       a8=PAR(9)
       a9=PAR(10)
       a10=PAR(12)

       F(1) = a1 + a2*x + a3*b + a4*x**2 + a5*b*x + a6*b**2 + &
              a7*x**3 + a8*b*x**2 + a9*b**2*x + a10*b**3
       
      END SUBROUTINE FUNC


      SUBROUTINE STPNT(NDIM,U,PAR,T)

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
