$("form[name=signup_form").submit(function(e) {
    var $form = $(this);
    var $error = $form.find(".error");
    var data = $form.serialize();

    $.ajax({
        url: "/user/signup",
        type: "POST",
        data: data,
        dataType: "json",
        success: function(res) {
            window.location.href = "/record/"; 
        },
        error: function(res) {
            console.log(res);
            $error.text(res.responseJSON.error).removeClass("error--hidden")
        }
    });

    e.preventDefault();
});

$("form[name=login_form").submit(function(e) {
    var $form = $(this);
    var $error = $form.find(".error");
    var data = $form.serialize();

    $.ajax({
        url: "/user/login",
        type: "POST",
        data: data,
        dataType: "json",
        success: function(res) {
            window.location.href = "/record/"; 
        },
        error: function(res) {
            console.log(res);
            $error.text(res.responseJSON.error).removeClass("error--hidden")
        }
    });

    e.preventDefault();
});

$(".record").on("click",function() {
    $(this).css("background-color", "red");
    $(this).text("Recording");
    $(".t").removeClass("t");
});
