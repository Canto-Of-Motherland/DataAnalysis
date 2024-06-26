function setBackgroundSize() {
    const screenWidth = $(window).width() - 400;
    const screenHeight = $(window).height();

    if (screenWidth / 2 > screenHeight / 1.13) {
        $('.background_img').css('width', '100%');
        $('.background_img').css('height', 'auto');
    } else {
        $('.background_img').css('height', '100%');
        $('.background_img').css('width', 'auto');
    }
}

$(document).ready(function() {
    setBackgroundSize();
});

$(window).on('resize', function() {
    setBackgroundSize();
});

function activeInput(obj) {
    let label = $(obj).siblings('.labels');
    let input = $(obj).siblings('.inputs');
    label.css({
        'z-index': '4',
        'color': '#1a315c',
        'font-size': '10px',
        'top': '-8px',
        'transition-duration': '300ms',
    });
    input.css({
        'border': '#1a315c 2px solid',
        'outline': 'none',
        'transition-duration': '300ms'
    });
}

function remainInput(obj) {
    let content = $(obj).val();
    if (content === '') {
        let label = $(obj).siblings('.labels');
        let input = $(obj).siblings('.inputs');
        label.css({
            'z-index': '2',
            'color': '#919191',
            'font-size': '15px',
            'top': '12px',
            'transition-duration': '300ms',
        });
        input.css({
            'border': '#6f6f6f 2px solid',
            'outline': 'none',
            'transition-duration': '300ms'
        });
    } else {
        let label = $(obj).siblings('.labels');
        let input = $(obj).siblings('.inputs');
        label.css({
            'z-index': '4',
            'color': '#919191',
            'font-size': '10px',
            'top': '-8px',
            'transition-duration': '300ms',
        });
        input.css({
            'border': '#6f6f6f 2px solid',
            'outline': 'none',
            'transition-duration': '300ms'
        });
    }
}

function toRight() {
    let element = $('.to_sign_code');
    if (!element.hasClass('active')) {
        element.addClass('active');
        $('.to_sign_password').removeClass('active');
        $('.sign').css({
            'transform': 'translateX(-440px)',
            'transition-duration': '300ms',
        });
    }
}

function toLeft() {
    let element = $('.to_sign_password');
    if (!element.hasClass('active')) {
        element.addClass('active');
        $('.to_sign_code').removeClass('active');
        $('.sign').css({
            'transform': 'translateX(-40px)',
            'transition-duration': '300ms',
        });
    }
}

$(document).ready(function() {
    toLeft();
    const inputEmail1 = '.input_email_1';
    const inputPassword = '.input_password';
    const inputEmail2 = '.input_email_2';
    const inputCode = '.input_code';
    remainInput(inputEmail1);
    remainInput(inputPassword);
    remainInput(inputEmail2);
    remainInput(inputCode);
});



function checkEmailInfo1() {
    let email = $('.input_email_1').val();
    const patternEmail = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
    let matchEmail = email.match(patternEmail);
    if (email === '') {
        $('.info_email_1').text('邮箱不能为空');
        $('.info_email_1').show();
    } else if (!matchEmail) {
        $('.info_email_1').text('邮箱格式不符');
        $('.info_email_1').show();
    } else {
        $('.info_email_1').hide();
    }
}

function checkPasswordInfo() {
    let password = $('.input_password').val();
    if (password === '') {
        $('.info_password').text('密码不能为空');
        $('.info_password').show();
    } else {
        $('.info_password').hide();
    }
}

function checkEmailInfo2() {
    let email = $('.input_email_2').val();
    const patternEmail = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
    let matchEmail = email.match(patternEmail);
    if (email === '') {
        $('.info_email_2').text('邮箱不能为空');
        $('.info_email_2').show();
    } else if (!matchEmail) {
        $('.info_email_2').text('邮箱格式不符');
        $('.info_email_2').show();
    } else {
        $('.info_email_2').hide();
    }
}

function checkCodeInfo() {
    let code = $('.input_code').val();
    if (code === '') {
        $('.info_code').text('验证码不能为空');
        $('.info_code').show();
    } else if (code.length !== 6) {
        $('.info_code').text('验证码位数应为6');
        $('.info_code').show();
    } else {
        $('.info_code').hide();
    }
}

function sendMail() {
    $('.btn_get_code').css('pointer-events', 'none');
    let email = $('.input_email_2').val();
    const patternEmail = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
    let matchEmail = email.match(patternEmail);
    if (email === '') {
        $('.info_email_2').text('邮箱不能为空');
        $('.info_email_2').show();
    } else if (!matchEmail) {
        $('.info_email_2').text('邮箱格式错误');
        $('.info_email_2').show();
    } else {
        $('.info_email_2').hide();

        let formData = new FormData();

        formData.append('csrfmiddlewaretoken', $('input[name="csrfmiddlewaretoken"]').val());
        formData.append('func_code', '1');
        formData.append('email', email);

        $.ajax({
            type: 'post',
            url: '/sign-in/',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                if (response['status_code'] === 0) {
                    console.log(response);
                    $('.info_email_2').text('发送成功，五分钟内有效');
                    $('.info_email_2').show();

                    let time = 60;
                    let timer;

                    $('.btn_get_code').text('已发送 (' + time + ')');
                    time = time - 1;
                    timer = setInterval(function () {
                        if (time > 0) {
                            $('.btn_get_code').text('已发送 (' + time + ')');
                            time = time - 1;
                            $('.btn_get_code').css('pointer-events', 'none');
                        } else {
                            $('.btn_get_code').css('pointer-events', 'auto');
                            $('.btn_get_code').text('获取验证码');
                            clearInterval(timer);
                        }
                    }, 1000);
                } else if (response['status_code'] === 1) {
                    $('.btn_get_code').css('pointer-events'), 'auto';
                    $('.info_email_2').text('发送失败');
                    $('.info_email_2').show();
                } else if (response['status_code'] === 2) {
                    $('.btn_get_code').css('pointer-events'), 'auto';
                    $('.info_email_2').text('用户不存在');
                    $('.info_email_2').show();
                }
            }
        });
    }
}

function signIn() {
    const element = $('.to_sign_password');
    if (element.hasClass('active')) {
        let email = $('.input_email_1').val();
        let password = $('.input_password').val();
        const patternEmail = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
        let matchEmail = email.match(patternEmail);
        if (email === '') {
            $('.info_email_1').text('邮箱不能为空');
            $('.info_email_1').show();
        } else if (!matchEmail) {
            $('.info_email_1').text('邮箱格式错误');
            $('.info_email_1').show();
        } else if (password === '') {
            $('.info_password').text('密码不能为空');
        } else {
            $('.info_email_1').hide();
            $('.info_password').hide();

            let formData = new FormData();

            formData.append('csrfmiddlewaretoken', $('input[name="csrfmiddlewaretoken"]').val());
            formData.append('func_code', '2')
            formData.append('email', email);
            formData.append('password', password);

            $.ajax({
                type: 'post',
                url: '/sign-in/',
                data: formData,
                processData: false,
                contentType: false,
                success: function (response) {
                    if (response['status_code'] === 0) {
                        window.location.href = '/';
                    } else if (response['status_code'] === 1) {
                        $('.info_password').text('密码错误');
                        $('.info_password').show();
                    } else if (response['status_code'] === 2) {
                        $('.info_email_1').text('未注册，请先注册');
                        $('.info_email_1').show();
                    }
                }
            });
        }
    } else {
        let email = $('.input_email_2').val();
        let code = $('.input_code').val();
        const patternEmail = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
        let matchEmail = email.match(patternEmail);
        if (email === '') {
            $('.info_email_2').text('邮箱不能为空');
            $('.info_email_2').show();
        } else if (!matchEmail) {
            $('.info_email_2').text('邮箱格式错误');
            $('.info_email_2').show();
        } else if (code === '') {
            $('.info_code').text('验证码不能为空');
            $('.info_code').show();
        } else if (code.length !== 6) {
            $('.info_code').text('验证码需为六位');
            $('.info_code').show();
        } else {
            $('.info_email_2').hide();
            $('.info_code').hide();

            let formData = new FormData();

            formData.append('csrfmiddlewaretoken', $('input[name="csrfmiddlewaretoken"]').val());
            formData.append('func_code', '3');
            formData.append('email', email);
            formData.append('code', code);

            $.ajax({
                type: 'post',
                url: '/sign-in/',
                data: formData,
                processData: false,
                contentType: false,
                success: function (response) {
                    console.log(response['status_code'])
                    if (response['status_code'] === 0) {
                        window.location.href = '/';
                    } else if (response['status_code'] === 1) {
                        $('.info_code').text('验证码已失效');
                        $('.info_code').show();
                    } else if (response['status_code'] === 2) {
                        $('.info_code').text('验证码错误');
                        $('.info_code').show();
                    } else if (response['status_code'] === 3) {
                        $('.info_code').text('验证码已过期');
                        $('.info_code').show();
                    } else if (response['status_code'] === 4) {
                        $('.info_email_2').text('未注册，请先注册');
                        $('.info_email_2').show();
                    } else if (response['status_code'] === 5) {
                        $('.info_code').text('请先获取验证码');
                        $('.info_code').show();
                    }
                }
            });
        }
    }
}