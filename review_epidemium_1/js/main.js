
// Déclenchement du graph timeline
function check() {
	var top = $('#community').offset().top;
	var scrollTop = $(window).scrollTop();
	var windowHeight = $(window).height();

	if (windowHeight + scrollTop > top)
	{
		$( "#courbe-masque" ).animate({
			width: "100%"
		}, 5000 );
	}

};


//Déplacement des titres H2
function check_2()
{
	$( "h2" ).each(function( index ) {
	  
		var parent = $(this).parent().parent();
		var top = parent.offset().top;
		var limite = top+parent.height();
		var scrollTop = $(window).scrollTop();
		var windowHeight = $(window).height();
		var windowWidth = $(window).width();
		var topPadding = 50;
		
		var calcul = 0;
		
		 
		if (scrollTop > top - topPadding && windowWidth >= 1170) {
			if(scrollTop < parent.next().offset().top - 700 ){
				calcul = Math.round(scrollTop-top) + topPadding;
				$(this).parent().css({
					marginTop : calcul
				})
			}
		} else {
			$(this).parent().css({
				marginTop : 0
			})
		}
    
	});
};


//Resize de la fenêtre
function resize()
{
	if($(window).width()<768)
	{
		$('.courbe	img	').css('width',$(window).width()-60);
		$('.diagramme-timeline .fond .ligne').css('margin-bottom',$('.courbe	img	').height()/9)	;
	}
}


//Quand le document est prêt, on lance les différentes fonctions
$(document).ready(function() {
	resize();
	$('img[usemap]').rwdImageMaps();
	check();
	check_2();

	// quand on scroll
	$(window).scroll(function() {
		check();
		check_2();
	});

	// quand on redimensionne
	$(window).resize(function() {
		resize();
		check();
		check_2();
	  
	});
  
  
  
 
  
});
