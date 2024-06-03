function logout() {
    let csrf = $('[name=csrfmiddlewaretoken]').val();
    let formData = new FormData();
    formData.append('csrfmiddlewaretoken', csrf);
    $.ajax({
        type: 'POST',
        url: '/logout/',
        data: formData,
        contentType: false,
        processData: false,
        success: function () {
            window.location.href = '/';
        }
    });
}