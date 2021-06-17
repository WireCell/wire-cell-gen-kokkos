
namespace WireCell {
    namespace GenKokkos {

       struct GdData {
           double p_ct ;
           double t_ct ;
	   double p_sigma ;
	   double t_sigma ;
	   double charge ;
           };

       struct DBin {
           double minval ;
           double binsize ;
	   int  nbins ;
       } ;

    }

}
