function setBackgroundSize() {
    const screenWidth = $(window).width() - 400;
    const screenHeight = $(window).height();

    if (screenWidth / 3 > screenHeight / 2) {
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

function sendMail() {
    $('.btn_get_code').css('pointer-events', 'none');
    let email = $('.input_email').val();
    const patternEmail = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
    let matchEmail = email.match(patternEmail);
    if (email === '') {
        $('.info_email').text('邮箱不能为空');
        $('.info_email').show();
    } else if (!matchEmail) {
        $('.info_email').text('邮箱格式不符');
        $('.info_email').show();
    } else {
        $('.info_email').hide();

        let formData = new FormData();

        formData.append('csrfmiddlewaretoken', $('input[name="csrfmiddlewaretoken"]').val());
        formData.append('func_code', '1');
        formData.append('email', email);

        $.ajax({
            type: 'post',
            url: '/sign-up/',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                if (response['status_code'] === 0) {
                    $('.info_email').text('发送成功，五分钟内有效');
                    $('.info_email').show();
                    
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
                    $('.info_email').text('该邮箱已注册');
                    $('.info_email').show();
                } else if (response['status_code'] === 2) {
                    $('.btn_get_code').css('pointer-events'), 'auto';
                    $('.info_email').text('发送失败');
                    $('.info_email').show();
                }
            }
        })

    }
    
}

$(document).ready(function() {
    const inputEmail = '.input_email';
    const inputCode = '.input_code';
    const inputUsername = '.input_username';
    const inputPassword = '.input_password';
    const inputConfirm = '.input_confirm';

    remainInput(inputEmail);
    remainInput(inputCode);
    remainInput(inputUsername);
    remainInput(inputPassword);
    remainInput(inputConfirm);
});

function checkEmailInfo() {
    let email = $('.input_email').val();
    const patternEmail = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
    let matchEmail = email.match(patternEmail);
    if (email === '') {
        $('.info_email').text('邮箱不能为空');
        $('.info_email').show();
    } else if (!matchEmail) {
        $('.info_email').text('邮箱格式不符');
        $('.info_email').show();
    } else {
        $('.info_email').hide();
    }
}

function checkCodeInfo() {
    let code = $('.input_code').val();
    if (code === '') {
        $('.info_code').text('验证码不能为空');
        $('.info_code').show();
    } else if (code.length !== 6) {
        $('.info_code').text('验证码需为六位');
        $('.info_code').show();
    } else {
        $('.info_code').hide();
    }
}

function toRight() {
    let email = $('.input_email').val();
    let code = $('.input_code').val();
    const patternEmail = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
    let match_email = email.match(patternEmail);
    if (email === '') {
        $('.info_email').text('邮箱不能为空');
        $('.info_email').show();
    } else if (!match_email) {
        $('.info_email').text('邮箱格式错误');
        $('.info_email').show();
    } else if (code === '') {
        $('.info_code').text('密码不能为空');
    } else if (code.length !== 6) {
        $('.info_code').text('验证码需为六位');
        $('.info_code').show();
    } else {
        $('.info_email').hide();
        $('.info_password').hide();
        
        let formData = new FormData();

        formData.append('csrfmiddlewaretoken', $('input[name="csrfmiddlewaretoken"]').val())
        formData.append('func_code', '2');
        formData.append('code', code);

        $.ajax({
            type: 'POST',
            url: '/sign-up/',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                if (response['status_code'] === 0) {
                    $('.sign').css({
                        'transform': 'translateX(-440px)',
                        'transition-duration': '300ms',
                    });
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
                    $('.info_email').text('请先获取验证码');
                    $('.info_email').show();
                }
            } 
        })
    }
}

function checkUsernameInfo() {
    let username = $('.input_username').val();
    if (username === '') {
        $('.info_username').text('用户名不能为空');
        $('.info_username').show();
    } else if (username.length > 16) {
        $('.info_username').text('用户名不能超过16位');
        $('.info_username').show();
    } else {
        $('.info_username').hide();
    }
}

function checkPasswordInfo() {
    let password = $('.input_password').val();
    const patternPassword = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d@$!%*?&]+$/;
    let matchPassword = password.match(patternPassword);
    if (password === '') {
        $('.info_password').text('密码不能为空');
        $('.info_password').show();
    } else if (!matchPassword) {
        $('.info_password').text('密码需含数字与大小写字母');
        $('.info_password').show();
    } else if (password.length < 8 || password.length > 16) {
        $('.info_password').text('密码长度需在8-16位之间');
        $('.info_password').show();
    } else {
        $('.info_password').hide();
    }
}

function checkConfirmInfo() {
    let password = $('.input_password').val();
    let confirm = $('.input_confirm').val();
    if (confirm === '') {
        $('.info_confirm').text('确认密码不能为空');
        $('.info_confirm').show();
    } else if (password !== confirm) {
        $('.info_confirm').text('上下密码不一致');
        $('.info_confirm').show();
    } else {
        $('.info_confirm').hide();
    }
}

function signUp() {
    let username = $('.input_username').val();
    let password = $('.input_password').val();
    let confirm = $('.input_confirm').val();
    const patternPassword = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d@$!%*?&]+$/;
    let matchPassword = password.match(patternPassword);
    if (username === '') {
        $('.info_username').text('用户名不能为空');
        $('.info_username').show();
    } else if (username.length > 16) {
        $('.info_username').text('用户名不能超过16位');
        $('.info_username').show();
    } else if (password === '') {
        $('.info_password').text('密码不能为空');
        $('.info_password').show();
    } else if (!matchPassword) {
        $('.info_password').text('密码需含数字与大小写字母');
        $('.info_password').show();
    } else if (password.length < 8 || password.length > 16) {
        $('.info_password').text('密码长度需在8-16位之间');
        $('.info_password').show();
    } else if (confirm === '') {
        $('.info_confirm').text('确认密码不能为空');
        $('.info_confirm').show();
    } else if (password !== confirm) {
        $('.info_confirm').text('上下密码不一致');
        $('.info_confirm').show();
    } else {
        $('.info_username').hide();
        $('.info_password').hide();
        $('.info_confirm').hide();

        let formData = new FormData();

        formData.append('csrfmiddlewaretoken', $('input[name="csrfmiddlewaretoken"]').val());
        formData.append('func_code', '3')
        formData.append('username', username);
        formData.append('password', password);

        $.ajax({
            type: 'POST',
            url: '/sign-up/',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                if ((response['status_code']) === 0) {
                    Swal.fire({
                        title: '注册成功',
                        icon: 'success',
                        width: '300px',
                        timer: 1500,
                        timerProgressBar: true,
                        showConfirmButton: true,
                        confirmButtonText: '确　定',
                        customClass: {
                            confirmButton: 'swal-btn-confirm',
                            title: 'swal-title',
                        },
                        onBeforeOpen: () => {
                            timerInterval = setInterval(() => {
                                const content = Swal.getContent()
                                if (content) {
                                    const b = content.querySelector('b');
                                    if (b) {
                                    b.textContent = parseInt(Swal.getTimerLeft() / 1000);
                                    }
                                }
                            }, 100);
                        },
                        onClose: () => {
                            clearInterval(timerInterval);
                        }
                    }).then(() => {
                        window.location.assign('/sign-in/');
                    });
                }
            }
        });
    }


}
