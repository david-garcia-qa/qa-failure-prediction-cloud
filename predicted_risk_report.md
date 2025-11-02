# QA Failure Risk Report

This report ranks automated tests by predicted probability of failure in the next run.

- test_payment_3ds_flow (payment) => risk HIGH (0.79)
- test_checkout_card (payment) => risk HIGH (0.78)
- test_checkout_address (payment) => risk HIGH (0.73)
- test_rate_limiter_lockout (security) => risk HIGH (0.72)
- test_login_mfa (auth) => risk HIGH (0.72)
- test_resend_sms_code (auth) => risk MEDIUM (0.65)
- test_change_password (account) => risk MEDIUM (0.63)
- test_password_reset (auth) => risk MEDIUM (0.61)
- test_audit_log_written (security) => risk MEDIUM (0.60)
- test_update_profile (account) => risk MEDIUM (0.56)
- test_remove_item_cart (cart) => risk MEDIUM (0.53)
- test_add_to_cart (cart) => risk MEDIUM (0.52)

Generated automatically from historical execution data.
