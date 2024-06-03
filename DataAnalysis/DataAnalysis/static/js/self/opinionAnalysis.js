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

function switchPanel(obj) {
    $('.switch_item').css({
        'color': '#6d6d6d',
        'background-color': 'transparent'
    });
    $(obj).css({
        'color': '#ffffff',
        'background-color': '#060044'
    });
    classId = $(obj).attr('id');
    $('.btn').css({
        'display': 'none'
    });
    $('.btn_submit_' + classId).css({
        'display': 'block'
    });
    $('.operate_with_topic').css({
        'display': 'none'
    });
    $('.operate_with_file').css({
        'display': 'none'
    });
    $('.operate_with_' + classId).css({
        'display': 'block'
    });
}

function expand(obj) {
    if ($(obj).text() == '展开') {
        $('.cookie').css({
            'display': 'block'
        });
        $(obj).css({
            'top': '137px'
        });
        $(obj).text('收起');
        $('.operate_panel_content').css({
            'height': '100px'
        });
        $('.operate_panel').css({
            'height': '140px'
        });
    } else {
        $('.cookie').css({
            'display': 'none'
        });
        $(obj).css({
            'top': '97px'
        });
        $(obj).text('展开');
        $('.operate_panel_content').css({
            'height': '60px'
        });
        $('.operate_panel').css({
            'height': '100px'
        });
    }
}

function changePath(obj) {
    file = $(obj).prop('files')[0];
    if (file) {
        fileName = file.name;
        $('.replace_file_path').text(fileName);
    } else {
        $('.replace_file_path').text('未选择文件');
    }
}

function changeValueInt(obj, value) {
    $({count: 0}).animate({count: value}, {
        duration: 3000,
        step: function () {
            let element = $(obj);
            element.text(Math.floor(this.count));
        }
    });
}

function changeValueFloat(obj, value) {
    $({count: 0.0}).animate({count: value}, {
        duration: 3000,
        step: function () {
            let element = $(obj);
            element.text(Math.floor(this.count * 1000) / 1000);
        }
    });
}

function activeColor(obj) {
    $(obj).css({
        'background-color': '#060044',
        'transition-duration': '200ms'
    });
    $(obj).children('label').css({
        'color': '#ffffff',
        'transition-duration': '200ms'
    });
}

function returnColor(obj) {
    $(obj).css({
        'background-color': '#ffffff',
        'transition-duration': '200ms'
    });
    $(obj).find('.number').css({
        'color': '#060044',
        'transition-duration': '200ms'
    });
    $(obj).find('.number_title').css({
        'color': '#454545',
        'transition-duration': '200ms'
    });
}

function changeFigure(obj) {
    if ($(obj).text() === '看舆情走势') {
        $('.figure_block_2_2').css({
            'display': 'none',
            'transition-duration': '200ms'
        });
        $('.figure_block_2_3').css({
            'display': 'block',
            'transition-duration': '200ms'
        });
        $(obj).text('看极化走势');
    } else {
        $('.figure_block_2_2').css({
            'display': 'block',
            'transition-duration': '200ms'
        });
        $('.figure_block_2_3').css({
            'display': 'none',
            'transition-duration': '200ms'
        });
        $(obj).text('看舆情走势');
    }
}

function submit() {
    $('.cookie').css({
        'display': 'none'
    });
    $('.expand').css({
        'top': '97px'
    });
    $('.expand').text('展开');
    $('.operate_panel_content').css({
        'height': '60px'
    });
    $('.operate_panel').css({
        'height': '100px'
    });

    $('.info').css('display', 'block');

    $('html, body').animate({
        scrollTop: $(".figure_block").offset().top
    }, 500);

    let formData = new FormData();
    formData.append('csrfmiddlewaretoken', $('input[name="csrfmiddlewaretoken"]').val());
    $.ajax({
        url: '/opinion-analysis/',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            let number_1 = '.number_1';
            let number_2 = '.number_2';
            let number_3 = '.number_3';
            let number_4 = '.number_4';
            let number_5 = '.number_5';
            let number_6 = '.number_6';
        
            changeValueInt(number_1, 87063);
            changeValueInt(number_2, 244);
            changeValueInt(number_3, 4208);
            changeValueFloat(number_4, 0.081);
            changeValueFloat(number_5, 0.850);
            changeValueFloat(number_6, 0.146);
            Plotly.newPlot('figure_2_1', response['data']['data_2_1']);
            Plotly.newPlot('figure_2_2', response['data']['data_2_2']);
            Plotly.newPlot('figure_2_3', response['data']['data_2_3']);
        }
    });
}
