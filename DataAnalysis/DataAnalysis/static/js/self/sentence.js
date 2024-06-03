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

function tag(obj) {
    let sentence = $('.textarea_1').val();
    var model = $('.dropdown').val();
    let csrf = $('[name=csrfmiddlewaretoken]').val();
    let formData = new FormData();
    if (sentence === '') {
        $('.textarea_1').focus();
        return
    }
    formData.append('func_code', '1');
    formData.append('csrfmiddlewaretoken', csrf);
    formData.append('sentence', sentence);
    formData.append('model', model);

    $(obj).text('标记中...');
    $('.output_area_inner_1').empty();

    $.ajax({
        type: 'POST',
        url: '/sentence/',
        data: formData,
        contentType: false,
        processData: false,
        success: function (response) {
            let data = response['data']
            let index = 0
            let interval = setInterval(function () {
                if (index < data.length) {
                    let code = '<div class="fragment"><div class="word">' + data[index]['word'] + '</div><div class="tag">' + data[index]['tag'] + '</div></div>'
                    $('.output_area_inner_1').append(code);
                    index++;
                } else {
                    clearInterval(interval);
                    $(obj).text('开始标记');
                }
            })
        }
    });
}


function recognize(obj) {
    let sentence = $('.textarea_2').val();
    let csrf = $('[name=csrfmiddlewaretoken]').val();
    let formData = new FormData();
    if (sentence === '') {
        $('.textarea_2').focus();
        return
    }
    formData.append('func_code', '2');
    formData.append('csrfmiddlewaretoken', csrf);
    formData.append('sentence', sentence);

    $(obj).text('识别中...');
    $('.output_area_inner_2').empty();

    $.ajax({
        type: 'POST',
        url: '/sentence/',
        data: formData,
        contentType: false,
        processData: false,
        success: function (response) {
            $(obj).text('开始识别');
            let data = response['data']
            for(let i = 0; i < data.length; i++) {
                let code = '<div class="entity"><div class="entity_name">' + data[i]['entity'] + '</div><div class="entity_tag">' + data[i]['tag'] + '</div></div>'
                $('.output_area_inner_2').append(code);
            }
        }
    });
}
