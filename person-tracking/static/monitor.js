var index = 0;
var tempImg = new Image();
tempImg.onload = function(){
   appendImage();
}

var tryLoadImage = function( index ){
   tempImg.src = '/static/' + index + '.png?';
}

var appendImage = function(){
   var img = document.createElement('img');
   img.src = tempImg.src;
   document.body.appendChild( img )
   tryLoadImage( index++ )
}
tryLoadImage( index );